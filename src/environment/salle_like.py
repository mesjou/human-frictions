import random
from typing import Dict, Tuple

import numpy as np
from agents.bank import CentralBank
from agents.firm import Firm
from agents.household import HouseholdAgent
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from utils import rewards


class NewKeynesMarket(MultiAgentEnv):
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

        self.unemployment = 0.0
        self.firm: Firm = Firm()

        self.inflation = 0.0
        self.interest = 0.0
        self.central_bank: CentralBank = CentralBank()

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
        """Defines the logic of a step in the environment.

        1.) Agents supply labor and earn income.
        2.) Firms set prices as a markup.
        3.) Agents consume from their initial budget.
        4.) Firm learns
        5.) Agents earn dividends from firms.
        6.) Central bank sets interest rate
        7.) Agents earn interest on their not consumed income.

        :param actions: (Dict) The action contains the reservation wage of each agent and the fraction of their budget
            they want to consume.

        :return obs: (Dict) The observation of the agents. This includes average wage of the period,
            their budget, the inflation and interest rates.
        """

        # 1. - 4.
        wages, demand = self.parse_actions(actions)
        self.clear_labor_market(wages)
        self.clear_goods_market(demand)

        # 5. - 7.
        self.clear_dividends(self.firm.profit)
        self.clear_capital_market()

        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage": np.mean([wage for wage in wages.values()]),
                "budget": agent.budget,
                "inflation": self.inflation,
                "interest": self.interest,
                "unemployment": self.unemployment,
            }

        assert self.firm.labor_demand <= self.n_agents, "Labor demand cannot be satisfied from agents"

        return obs

    def compute_rewards(self):
        rew = {}
        for agent in self.agents.values():
            rew[agent.agent_id] = rewards.utility(labor=agent.labor, consumption=agent.consumption)
        return rew

    def parse_actions(self, actions):
        wages = {}
        consumption = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            consumption[agent.agent_id] = agent_action[0]
            wages[agent.agent_id] = agent_action[1]
        return wages, consumption

    def clear_labor_market(self, wages):
        """Household can supply labor and firms decide which to hire"""
        occupation = self.firm.hire_worker(wages)
        for agent in self.agents.values():
            agent.earn(occupation[agent.agent_id])
        self.firm.produce(occupation)
        self.inflation = self.firm.set_price(occupation, wages)
        self.unemployment = (self.n_agents - self.firm.labor_demand) / self.n_agents
        self.firm.learn(self.n_agents)

    def clear_goods_market(self, demand):
        """Household wants to buy goods from the firm

        :param demand: (dict) defines how much each agent wants to consume in real values
        """
        consumption = self.firm.sell_goods(demand)
        for agent in self.agents.values():
            agent.consume(consumption[agent.agent_id], self.firm.price)
        self.firm.earn_profits(consumption)

    def clear_dividends(self, profit):
        """Each agent receives a dividend computed as the share of total profit divided by number of agents"""
        for agent in self.agents.values():
            agent.budget += profit / self.n_agents

    def clear_capital_market(self):
        """Agents earn interest on their budget balance which is specified by central bank"""
        self.interest = self.central_bank.set_interest_rate(unemployment=self.unemployment, inflation=self.inflation)
        for agent in self.agents.values():
            agent.budget += self.interest * agent.budget
