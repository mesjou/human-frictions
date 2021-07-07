import random
from typing import Dict, Tuple

import numpy as np
from agents.household import HouseholdAgent
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from utils import rewards
from utils.firm import Firm


class SalleLike(MultiAgentEnv):
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

        init_budget = config["init_budget"]
        assert isinstance(init_budget, float)
        self.init_budget = init_budget

        self.timestep = 0
        self.agents: Dict[int:HouseholdAgent] = {}
        self.firm: Firm = Firm

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

    def set_up_agents(self):
        """Initialize the agents and give them starting endowment."""
        for idx in range(self.n_agents):
            agent_id = "agent-" + str(idx)
            agent = HouseholdAgent(agent_id, self.init_budget)
            self.agents[agent_id] = agent

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        """
        self.timestep = 0
        self.agents = {}
        self.setup_agents()
        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage": 0.0,
                "budget": agent.budget,
                "inflation": 0.0,
                "interest": 0.0,
            }
        return obs

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.timestep += 1

        obs = self.generate_observations(actions)
        rew = self.compute_rewards()
        done = {"__all__": self.timestep >= self._episode_length}
        info = {}

        return obs, rew, done, info

    def generate_observations(self, actions):

        # supply labor via wages
        wages, fraction_consumption = self.parse_actions(actions)
        self.supply_labor(wages)
        self.consume(fraction_consumption)

        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage": 0.0,
                "budget": agent.budget,
                "inflation": 0.0,
                "interest": 0.0,
            }
        return obs

    def compute_rewards(self):
        rew = {}
        for agent in self.agents.values():
            rew[agent.agent_id] = rewards.utility(labor=agent.labor, consumption=agent.consumption,)
        return rew

    def parse_actions(self, actions):
        wages = {}
        consumption = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            consumption[agent.agent_id] = agent_action[0]
            wages[agent.agent_id] = agent_action[1]
        return wages, consumption

    def supply_labor(self, wages):
        """Household can supply labor and firms decide which to hire"""
        # occupation = self.firm.hire_worker(wages)
        raise NotImplementedError

    def consume(self, fraction_consumption):
        for agent in self.agents:
            raise NotImplementedError
