import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from ray.rllib.utils.typing import MultiAgentDict


class BaseEnv(ABC):
    def __init__(self, config):
        """
        Super class for environments that implements some basic functionality.

        Args:
            config (dict): A dictionary with configuration parameter {"parameter name": parameter value} specifying the
                parameter value if it is in the dictionary.
                n_agents (int): The number of household agents.
                episode_length (int): Number of timesteps in a single episode.
                seed (float, int): Seed value to use. Must be > 0.
        """

        n_agents = config.get("n_agents", None)
        assert isinstance(n_agents, int), "Number of agents must be specified as int in config['n_agents']"
        assert n_agents >= 1
        self.n_agents = n_agents

        episode_length = config.get("episode_length", None)
        assert isinstance(episode_length, int), "Episode length must be specified as int in config['episode_length']"
        assert episode_length >= 1
        self._episode_length = episode_length

        seed = config.get("seed", None)
        if seed is not None:
            self.seed(seed)

        self.timestep: int = 0

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

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        pass
