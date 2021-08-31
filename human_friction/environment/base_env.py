import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple

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

        Args:
            seed (int/float): Seed value to use. Must be > 0. Converted to int
                internally if provided value is a float.
        """
        assert isinstance(seed, (int, float))
        seed = int(seed)
        assert seed > 0

        np.random.seed(seed)
        random.seed(seed)

    def reset(self) -> MultiAgentDict:
        """Set up the environment to init values specified in config and set up agents.

        Returns:
            MultiAgentDict: dict containing a first observation for each agent {agent_id: obs}

        """
        self.timestep = 0
        self.set_up_agents()
        wages, demands = self.reset_env()

        obs = self.generate_observations(wages, demands)

        return obs

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Take an environment step.

        First update timestep and the take action, generate observation and compute rewards.

        Args:
            actions (MultiAgentDict): action for each agent {agent_id: action}

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]: observation, rewards, done and info
                for all agents and the environment

        """
        self.timestep += 1

        wages, demands = self.take_actions(actions)
        obs = self.generate_observations(wages, demands)
        rew = self.compute_rewards()
        done = {"__all__": self.timestep >= self._episode_length}
        info = self.generate_info()

        return obs, rew, done, info

    @abstractmethod
    def reset_env(self):
        """Take init values and reset all environment entities, e.g. firm and central bank"""
        pass

    @abstractmethod
    def set_up_agents(self):
        """Take init values and reset all acting agents."""
        pass

    @abstractmethod
    def take_actions(self, actions: MultiAgentDict):
        """Implements what happens when each agent takes her action.

        Args:
            actions (MultiAgentDict): Dict containing the action for each agent.

        """
        pass

    @abstractmethod
    def generate_observations(self, wages: MultiAgentDict, demands: MultiAgentDict) -> MultiAgentDict:
        """Generates the observation of the environment.

        Returns:
            MultiAgentDict: Dict with observation for each agent {agent_id: observation}

        """
        pass

    @abstractmethod
    def compute_rewards(self) -> MultiAgentDict:
        """Compute the reward for each agent.

        Returns:
            MultiAgentDict: Dict with reward for each agent {agent_id: reward}

        """
        pass

    def generate_info(self) -> Dict:
        """This function can be used to return additional information at each step."""
        return {}

    def get_custom_metrics(self) -> Dict:
        """Generate metrics for each step of the environment.

        This could be agent budget, wages consumption etc. The metrics will be fetched by
        rllib callbacks and can be visualized in Tensorboard.

        Returns:
            dict: a dictionary of {metric_key: value} where 'value' is a scalar.

        """
        return {}
