from typing import Tuple

from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class BaseEnv(MultiAgentEnv):
    def __init__(self, n_agents):
        """

        :param n_agents:
        """

        assert isinstance(n_agents, int)
        assert n_agents >= 1
        self.n_agents = n_agents

    def reset(self):
        obs = {}
        return obs

    def step(self, action: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs = {}
        rewards = {}
        done = {}
        info = {}
        return obs, rewards, done, info
