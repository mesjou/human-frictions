import math
from typing import Tuple

import numpy as np
from human_friction.agents.nonprofitfirm import SimpleFirm
from human_friction.environment.new_keynes import NewKeynesMarket
from human_friction.utils.annotations import override
from ray.rllib.utils.typing import MultiAgentDict


class SimpleNewKeynes(NewKeynesMarket):
    def __init__(self, config):
        technology = config.get("technology", 1.0)
        assert technology > 0.0
        assert isinstance(technology, float)

        alpha = config.get("alpha", 0.25)
        assert isinstance(alpha, float)
        assert 0.0 <= alpha < 1.0

        super().__init__(config)
        self.firm: SimpleFirm = SimpleFirm(
            technology=technology, alpha=alpha,
        )

    def get_max_consumption(self):
        return self.firm.technology * self.n_agents ** (-self.firm.alpha)

    def reset(self):
        # todo make this better!
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        """
        self.timestep = 0
        self.agents = {}
        self.set_up_agents()
        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage_increase": 0.02,
                "average_consumption": 0.05,
                "budget": self.init_budget,
                "inflation": 0.01,
                "employed_hours": 0.9,
                "interest": 0.02,
                "unemployment": 0.01,
                "action_mask": self.get_action_mask(agent),
            }
        return obs

    @override(NewKeynesMarket)
    def generate_observations(self, actions: MultiAgentDict) -> MultiAgentDict:
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

        # 1. - 3.
        wage_increases, demand = self.parse_actions(actions)
        wages = {agent.agent_id: agent.wage * (1 + wage_increases[agent.agent_id]) for agent in self.agents.values()}
        self.clear_markets(demand, wages)

        # 4. - 5.
        self.clear_capital_market()

        obs = {}

        for agent in self.agents.values():

            obs[agent.agent_id] = {
                "average_wage_increase": np.mean([wage_increases[agent.agent_id] for agent in self.agents.values()]),
                "average_consumption": np.mean([demand[agent.agent_id] for agent in self.agents.values()]),
                "budget": agent.budget,
                "inflation": self.inflation,
                "employed_hours": agent.labor,
                "interest": self.interest,
                "unemployment": self.unemployment,
                "action_mask": self.get_action_mask(agent),
            }

        assert self.firm.labor_demand <= self.n_agents, "Labor demand cannot be satisfied from agents"

        return obs

    @override(NewKeynesMarket)
    def parse_actions(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict]:
        wages = {}
        consumptions = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            wage, consumption = self.map_action_to_values(agent_action)
            wages[agent.agent_id] = wage
            consumptions[agent.agent_id] = consumption
        return wages, consumptions

    def map_action_to_values(self, action_idx, n_c_actions=10, n_w_actions=5):

        c_index = math.floor(action_idx / n_w_actions)
        w_index = action_idx - c_index * n_w_actions

        consumption = np.linspace(0.01, self.get_max_consumption(), n_c_actions)[c_index]
        wage_increase = np.linspace(0.0, 0.1, n_w_actions)[w_index]

        return wage_increase, consumption

    def clear_markets(self, demand: MultiAgentDict, wages: MultiAgentDict):
        """Household wants to buy goods from the firm

        Args:
            demand: (MultiAgentDict) defines how much each agent wants to consume in real values
            wages: (MultiAgentDict) defines how much each agent wants to earn in nominal values
        """
        self.firm.produce(demand)
        self.firm.get_labor_demand()
        occupation = self.firm.hire_worker(wages)
        self.inflation = self.firm.set_price(occupation, wages)
        self.unemployment = self.get_unemployment()

        for agent in self.agents.values():
            agent.earn(occupation[agent.agent_id], wages[agent.agent_id])
            agent.consume(demand[agent.agent_id], self.firm.price)

    def get_action_mask(self, agent):
        actions = np.zeros(50)
        max_c = agent.budget / self.firm.price
        n = self.scale(max_c)
        actions[0 : int(n)] = 1.0  # noqa E203
        return actions

    def scale(self, old_value, n_c_actions=10, n_w_actions=5):
        if old_value > self.get_max_consumption():
            new_value = n_c_actions
        else:
            old_range = self.get_max_consumption() - 0.01
            new_range = n_c_actions - 0.0
            new_value = ((old_value - 0.01) * new_range) / old_range + 0.0
        return max(1, math.floor(new_value)) * n_w_actions
