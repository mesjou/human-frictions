import random
from typing import Dict, Tuple

import numpy as np
from human_friction.agents.bank import CentralBank
from human_friction.agents.firm import Firm
from human_friction.agents.household import HouseholdAgent
from human_friction.utils import rewards
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class NewKeynesMarket(MultiAgentEnv):
    """
    New Keynes Environment class. Should be used to simulate labor supply and consume decision of households.
    Instantiates the households, firm and central bank.

    Args:
        config (dict): A dictionary with configuration parameter {"parameter name": parameter value} specifying the
            parameter value if it is in the dictionary.

            The dictionary keys must include:
                n_agents (int): The number of household agents.
                episode_length (int): Number of timesteps in a single episode.

            The dictionary keys could include:
                Agent specific
                init_budget (float): How much budget each household agent should have at beginning of an episode.

                Env specific
                init_unemployment (float): How much unemployment is at the beginning of an episode, default = 0.0.
                init_inflation (float): How much inflation is at the beginning of an episode, default = 0.02.
                init_interest (float): How much nominal interest is at the beginning of an episode, default = 1.02.

                Firm specific
                technology (float): The technology factor of the firm, default = 0.5.
                alpha (float): Output elasticity of the firm`s Cobb-Douglas production function, default = 0.25.
                learning_rate (float): How fast does the firm updates its labor demand, default = 0.01.
                markup (float): The firm's price markup on its marginal costs, default = 0.1.
                memory (float): The firm's memory of past profits, determines how much weight is given to past profits
                    during learning, default = 0.45.

                Central Bank specific
                inflation_target (float): Inflation target of the central bank, default = 0.02.
                natural_unemployment (float): The natural unemployment of the economy, default = 0.0.
                natural_interest (float): The natural interest of the economy, default = 0.0.
                phi_unemployment (float): The CB's reaction coefficients to unemployment, default = 0.1.
                phi_inflation (float): The CB's reaction coefficients to inflation, default = 0.2.
    """

    def __init__(self, config):

        # Non-optional parameters
        # ----------

        n_agents = config.get("n_agents", None)
        assert isinstance(n_agents, int), "Number of agents must be specified as int in config['n_agents']"
        assert n_agents >= 1
        self.n_agents = n_agents

        episode_length = config.get("episode_length", None)
        assert isinstance(episode_length, int), "Episode length must be specified as int in config['episode_length']"
        assert episode_length >= 1
        self._episode_length = episode_length

        # Optional parameters
        # ----------

        init_budget = config.get("init_budget", 0.0)
        assert isinstance(init_budget, float)
        self.init_budget = init_budget

        init_unemployment = config.get("init_unemployment", 0.0)
        assert isinstance(init_unemployment, float)
        assert 0.0 <= init_unemployment <= 1.0
        self.unemployment = init_unemployment

        init_inflation = config.get("init_inflation", 0.02)
        assert isinstance(init_inflation, float)
        self.inflation = init_inflation

        init_interest = config.get("init_interest", 1.02)
        assert isinstance(init_interest, float)
        self.interest = init_interest

        technology = config.get("technology", 0.5)
        assert technology > 0.0
        assert isinstance(technology, float)

        alpha = config.get("alpha", 0.25)
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        learning_rate = config.get("learning_rate", 0.01)
        assert isinstance(learning_rate, float)
        assert learning_rate > 0.0

        markup = config.get("markup", 0.1)
        assert isinstance(markup, float)
        assert markup >= 0.0

        memory = config.get("memory", 0.45)
        assert isinstance(memory, float)
        assert memory > 0.0

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

        # Final setup of the environment
        # ----------

        self.timestep: int = 0
        self.agents: Dict[str:HouseholdAgent] = {}
        self.firm: Firm = Firm(
            init_labor_demand=float(self.n_agents),
            technology=technology,
            alpha=alpha,
            learning_rate=learning_rate,
            markup=markup,
            memory=memory,
        )
        self.central_bank: CentralBank = CentralBank(
            inflation_target=inflation_target,
            natural_unemployment=natural_unemployment,
            natural_interest=natural_interest,
            phi_unemployment=phi_unemployment,
            phi_inflation=phi_inflation,
        )

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
        self.set_up_agents()
        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "average_wage": 0.0,
                "budget": self.init_budget,
                "inflation": 0.0,
                "interest": 1.0,
                "unemployment": 0.0,
            }
        return obs

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.timestep += 1

        obs = self.generate_observations(actions)
        rew = self.compute_rewards()
        done = {"__all__": self.timestep >= self._episode_length}
        info = {}

        return obs, rew, done, info

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

    def compute_rewards(self) -> MultiAgentDict:
        rew = {}
        for agent in self.agents.values():
            rew[agent.agent_id] = rewards.utility(labor=agent.labor, consumption=agent.consumption)
        return rew

    def parse_actions(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict]:
        wages = {}
        consumption = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            consumption[agent.agent_id] = agent_action[0]
            wages[agent.agent_id] = agent_action[1]
        return wages, consumption

    def clear_labor_market(self, wages: MultiAgentDict):
        """Household can supply labor and firms decide which to hire"""
        occupation = self.firm.hire_worker(wages)
        for agent in self.agents.values():
            agent.earn(occupation[agent.agent_id], wages[agent.agent_id])
        self.firm.produce(occupation)
        self.inflation = self.firm.set_price(occupation, wages)
        self.unemployment = self.get_unemployment()

    def clear_goods_market(self, demand: MultiAgentDict):
        """Household wants to buy goods from the firm

        :param demand: (dict) defines how much each agent wants to consume in real values
        """
        consumption = self.firm.sell_goods(demand)
        for agent in self.agents.values():
            agent.consume(consumption[agent.agent_id], self.firm.price)
        self.firm.earn_profits(consumption)
        self.firm.learn(self.n_agents)
        self.firm.update_average_profit()

    def clear_dividends(self, profit: float):
        """Each agent receives a dividend computed as the share of total profit divided by number of agents"""
        for agent in self.agents.values():
            agent.budget += profit / self.n_agents

    def clear_capital_market(self):
        """Agents earn interest on their budget balance which is specified by central bank"""
        self.interest = self.central_bank.set_interest_rate(unemployment=self.unemployment, inflation=self.inflation)

        # assert self.interest >= 1.0, "Negative interest is not allowed"
        for agent in self.agents.values():
            agent.budget = self.interest * agent.budget

    def get_unemployment(self):
        assert self.firm.labor_demand > 0.0
        return (self.n_agents - self.firm.labor_demand) / self.n_agents
