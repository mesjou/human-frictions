import math
from typing import Dict, Tuple

import numpy as np
from human_friction.agents.bank import CentralBank
from human_friction.agents.household import HouseholdAgent
from human_friction.agents.nonprofitfirm import SimpleFirm
from human_friction.environment.base_env import BaseEnv
from human_friction.rewards import rewards
from human_friction.utils.annotations import override
from ray.rllib.utils.typing import MultiAgentDict


class SimpleNewKeynes(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        # Additional parameters of the environment and the agents
        # ----------

        labor_coefficient = config.get("labor_coefficient", 0.0)
        assert isinstance(labor_coefficient, float)
        assert labor_coefficient >= 0.0
        self.labor_coefficient = labor_coefficient

        technology = config.get("technology", 1.0)
        assert technology > 0.0
        assert isinstance(technology, float)

        alpha = config.get("alpha", 0.25)
        assert isinstance(alpha, float)
        assert 0.0 <= alpha < 1.0

        inflation_target = config.get("inflation_target", 0.02)
        assert isinstance(inflation_target, float)

        natural_unemployment = config.get("natural_unemployment", 0.0)
        assert isinstance(natural_unemployment, float)
        assert 1.0 >= natural_unemployment >= 0.0

        natural_interest = config.get("natural_interest", 0.0)
        assert isinstance(natural_interest, float)

        phi_unemployment = config.get("phi_unemployment", 0.1)
        assert isinstance(phi_unemployment, float)
        assert phi_unemployment > 0.0

        phi_inflation = config.get("phi_inflation", 0.2)
        assert isinstance(phi_inflation, float)
        assert phi_inflation > 0.0

        # Initial values
        # ----------

        init_budget = config.get("init_budget", 0.0)
        assert isinstance(init_budget, float)
        self.init_budget = init_budget

        init_wage = config.get("init_wage", 0.5623413251903491)
        assert isinstance(init_wage, float)
        assert init_wage > 0.0
        self.init_wage = init_wage

        init_unemployment = config.get("init_unemployment", 0.0)
        assert isinstance(init_unemployment, float)
        assert 0.0 <= init_unemployment < 1.0
        self.init_unemployment = init_unemployment

        init_inflation = config.get("init_inflation", 0.02)
        assert isinstance(init_inflation, float)
        self.init_inflation = init_inflation

        init_interest = config.get("init_interest", 1.02)
        assert isinstance(init_interest, float)
        self.init_interest = init_interest

        # Add entities to environment
        # ----------
        self.unemployment = 0.0
        self.inflation = 0.0
        self.interest = 0.0
        self.agents: Dict[str:HouseholdAgent] = {}
        self.firm: SimpleFirm = SimpleFirm(
            technology=technology, alpha=alpha,
        )
        self.central_bank: CentralBank = CentralBank(
            inflation_target=inflation_target,
            natural_unemployment=natural_unemployment,
            natural_interest=natural_interest,
            phi_unemployment=phi_unemployment,
            phi_inflation=phi_inflation,
        )

    @override(BaseEnv)
    def set_up_agents(self):
        """Initialize the agents and give them starting endowment."""
        for idx in range(self.n_agents):
            agent_id = "agent-" + str(idx)
            agent = HouseholdAgent(agent_id, self.init_budget, self.init_wage)
            self.agents[agent_id] = agent

    @override(BaseEnv)
    def reset_env(self):
        """Take init values and reset all environment entities, e.g. firm and central bank"""
        self.unemployment = self.init_unemployment
        self.inflation = self.init_inflation
        self.interest = self.init_interest

        labor_demand = (1 - self.unemployment) * self.n_agents
        production = self.firm.production_function(labor_demand)
        labor_costs = self.init_wage * labor_demand
        price = labor_costs / production

        self.firm.reset(
            price=price, production=production, labor_costs=labor_costs, labor_demand=labor_demand,
        )

        wage_increases = {}
        demand = {}
        consume = production / self.n_agents
        for agent_id, agent in self.agents.items():
            agent.reset(labor=1 - self.init_unemployment, consumption=consume)
            wage_increases[agent_id] = self.inflation
            demand[agent_id] = consume

        return wage_increases, demand

    @override(BaseEnv)
    def take_actions(self, actions: MultiAgentDict):
        """Defines the logic of a step in the environment.

        1.) Agents supply labor and earn income.
        2.) Firms set price to have zero profit.
        3.) Agents consume from their budget.
        4.) Central bank sets interest rate
        5.) Agents earn interest on their not consumed income.

        Args
            actions: (Dict) The action contains the wage increase of each agent and the real value they want to consume.

        """

        # 1. - 3.
        wage_increases, demand = self.parse_actions(actions)
        wages = {agent.agent_id: agent.wage * (1 + wage_increases[agent.agent_id]) for agent in self.agents.values()}
        self.clear_markets(demand, wages)

        # 4. - 5.
        self.clear_capital_market()

        return wage_increases, demand

    @override(BaseEnv)
    def generate_observations(self, wage_increases: MultiAgentDict, demands: MultiAgentDict) -> MultiAgentDict:

        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage_increase": np.mean([wage_increases[agent.agent_id] for agent in self.agents.values()]),
                "average_consumption": np.mean([demands[agent.agent_id] for agent in self.agents.values()]),
                "budget": agent.budget / self.firm.price,
                "inflation": self.inflation,
                "employed_hours": agent.labor,
                "interest": self.interest,
                "unemployment": self.unemployment,
                "action_mask": self.get_action_mask(agent),
            }

        return obs

    @override(BaseEnv)
    def generate_info(self):
        info = {}
        for agent in self.agents.values():
            info[agent.agent_id] = {
                "interets": self.interest,
                "end_of_period_budget": agent.budget,
                "actual_consumption": agent.consumption,
            }
        return info

    def clear_markets(self, demand: MultiAgentDict, wages: MultiAgentDict):
        """Household wants to buy goods from the firm

        Args:
            demand: (MultiAgentDict) defines how much each agent wants to consume in real values
            wages: (MultiAgentDict) defines how much each agent wants to earn in nominal values
        """
        self.firm.produce(demand)
        self.firm.get_labor_demand(self.n_agents)
        occupation = self.firm.hire_worker(wages)
        self.inflation = self.firm.set_price(occupation, wages)
        self.unemployment = self.get_unemployment()

        for agent in self.agents.values():
            agent.earn(occupation[agent.agent_id], wages[agent.agent_id])
            agent.consume(demand[agent.agent_id], self.firm.price)

    def clear_capital_market(self):
        """Agents earn interest on their budget balance which is specified by central bank"""
        self.interest = self.central_bank.set_interest_rate(unemployment=self.unemployment, inflation=self.inflation)

        assert self.interest >= 1.0, "Negative interest is not allowed"
        for agent in self.agents.values():
            agent.budget = self.interest * agent.budget

    def get_unemployment(self):
        assert self.firm.labor_demand > 0.0
        return (self.n_agents - self.firm.labor_demand) / self.n_agents

    def get_action_mask(self, agent: HouseholdAgent) -> np.array:
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
        return max(1, math.ceil(new_value)) * n_w_actions

    def parse_actions(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict]:
        wages = {}
        consumptions = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            wage, consumption = self.map_action_to_values(agent_action)
            wages[agent.agent_id] = wage
            consumptions[agent.agent_id] = consumption
        return wages, consumptions

    def get_max_consumption(self):
        total_production = self.firm.production_function(float(self.n_agents))
        return total_production / self.n_agents

    def map_action_to_values(self, action_idx, n_c_actions=10, n_w_actions=5):

        c_index = math.floor(action_idx / n_w_actions)
        w_index = action_idx - c_index * n_w_actions

        consumption = np.linspace(0.01, self.get_max_consumption(), n_c_actions)[c_index]
        wage_increase = np.linspace(0.0, 0.1, n_w_actions)[w_index]

        return wage_increase, consumption

    @override(BaseEnv)
    def compute_rewards(self) -> MultiAgentDict:
        rew = {}
        for agent in self.agents.values():
            rew[agent.agent_id] = rewards.utility(
                labor=agent.labor, consumption=agent.consumption, labor_coefficient=self.labor_coefficient
            )
        return rew

    @override(BaseEnv)
    def get_custom_metrics(self):
        """
        Generate metrics for each step of the environment. This could be agent budget, wages consumption etc.
        Returns (dict): return a dictionary of {metric_key: value} where 'value' is a scalar.

        """
        metrics = dict()
        example_agent = list(self.agents.values())[0]

        agent_quanities = [
            a for a in dir(example_agent) if not a.startswith("__") and not callable(getattr(example_agent, a))
        ]
        agent_quanities.remove("agent_id")

        for quantity in agent_quanities:
            entries = list()
            for agent in self.agents.values():
                entries.append(getattr(agent, quantity))
            metrics["agent_metrics/" + quantity] = np.mean(entries)

        action_mask = list()
        for agent in self.agents.values():
            action_mask.append(self.get_action_mask(agent).mean())
        metrics["agent_metrics/" + "action_mask"] = np.mean(action_mask)

        env_quanities = ["inflation", "interest", "unemployment"]
        for quantity in env_quanities:
            metrics["env_metrics/" + quantity] = getattr(self, quantity)

        metrics["env_metrics/" + "price"] = self.firm.price

        return metrics
