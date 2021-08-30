from human_friction.agents.learningfirm import LearningFirm
from human_friction.environment.simple_nk import SimpleNewKeynes
from human_friction.utils.annotations import override
from ray.rllib.utils.typing import MultiAgentDict


class NewKeynesMarket(SimpleNewKeynes):
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
                labor_coefficient (float): How much negative utility from working occurs.

                Env specific
                seed (float, int): Seed value to use. Must be > 0.
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

        super().__init__(config)

        # Additional parameters of the environment and the agents
        # ----------

        technology = config.get("technology", 1.0)
        assert technology > 0.0
        assert isinstance(technology, float)

        alpha = config.get("alpha", 0.25)
        assert isinstance(alpha, float)
        assert 0.0 <= alpha < 1.0

        learning_rate = config.get("learning_rate", 0.01)
        assert isinstance(learning_rate, float)
        assert learning_rate > 0.0

        markup = config.get("markup", 0.1)
        assert isinstance(markup, float)
        assert markup >= 0.0

        memory = config.get("memory", 0.45)
        assert isinstance(memory, float)
        assert memory > 0.0

        self.firm: LearningFirm = LearningFirm(
            technology=technology, alpha=alpha, learning_rate=learning_rate, markup=markup, memory=memory,
        )

    @override(SimpleNewKeynes)
    def reset_env(self):
        """Take init values and reset all environment entities, e.g. firm and central bank"""
        self.unemployment = self.init_unemployment
        self.inflation = self.init_inflation
        self.interest = self.init_interest

        labor_demand = (1 - self.unemployment) * self.n_agents
        production = self.firm.production_function(labor_demand)
        labor_costs = self.init_wage * labor_demand
        price = (1 + self.firm.markup) / (1 - self.firm.alpha) * labor_costs / production
        profit = price * production - labor_costs
        self.firm.reset(
            price=price,
            production=production,
            labor_costs=labor_costs,
            labor_demand=labor_demand,
            average_profit=profit,
            profit=profit,
        )

        wage_increases = {}
        demand = {}
        consume = production / self.n_agents
        for agent_id, agent in self.agents.items():
            agent.reset(labor=1 - self.init_unemployment, consumption=consume)
            wage_increases[agent_id] = self.inflation
            demand[agent_id] = consume

        return wage_increases, demand

    @override(SimpleNewKeynes)
    def take_actions(self, actions: MultiAgentDict):
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
        wage_increases, demand = self.parse_actions(actions)
        wages = {agent.agent_id: agent.wage * (1 + wage_increases[agent.agent_id]) for agent in self.agents.values()}
        self.clear_labor_market(wages)
        self.clear_goods_market(demand)

        # 5. - 7.
        self.clear_dividends(self.firm.profit)
        self.clear_capital_market()

        return wage_increases, demand

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

        Args:
            demand: (MultiAgentDict) defines how much each agent wants to consume in real values
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
