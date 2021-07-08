import random
from typing import Dict


class Firm(object):
    def __init__(self, technology: float = 0.5, alpha: float = 0.2, learning_rate: float = 0.1, markup: float = 0.1):
        assert isinstance(technology, float)
        assert technology >= 0.0
        self.technology = technology

        assert isinstance(alpha, float)
        assert alpha >= 0.0
        self.alpha = alpha

        assert isinstance(learning_rate, float)
        assert learning_rate >= 0.0
        self.learning_rate = learning_rate

        assert isinstance(markup, float)
        assert markup >= 0.0
        self.markup = markup

        self.labor_demand: float = 0.0
        self.average_profit: float = 0.0
        self.profit: float = 0.0
        self.price: float = 0.0
        self.production: float = 0.0
        self.labor_costs: float = 0.0

    def produce(self, occupation: Dict):
        labor = sum([occupation[agent.agent_id] for agent in occupation.keys()])
        assert labor >= 0.0
        self.production = self.technology * labor ** (1 - self.alpha)

    def learn(self):
        if self.profit / self.price >= self.average_profit:
            self.labor_demand = self.labor_demand * (1 + self.learning_rate)
        else:
            self.labor_demand = self.labor_demand * (1 - self.learning_rate)

    def hire_worker(self, wages: Dict):
        """Determine the hiring status based on reservation wages.

        The firm hires agents based on the wages. Lowest wages are hired first until labor demand is satisfied.
        """
        sorted_wages = {k: v for k, v in sorted(wages.items(), key=lambda item: item[1])}
        labor_demand = {}
        hired_labor = 0.0
        for agent_id, wage in sorted_wages.items():
            if (self.labor_demand - hired_labor) >= 1:
                demand = 1.0
            elif 1.0 > (self.labor_demand - hired_labor) > 0.0:
                demand = self.labor_demand - hired_labor
            else:
                demand = 0.0
            labor_demand[agent_id] = demand
            hired_labor += demand

        return labor_demand

    def set_price(self, occupation, wages):
        self.labor_costs = sum([occupation[agent.agent_id] * wages[agent.agent_id] for agent in wages.keys()])
        self.price = (1 + self.markup) / (1 - self.alpha) * self.labor_costs / self.production

    def sell_goods(self, demand: Dict):
        """Determine how much the firm sells to which agent.

        The order of the selling process is randomly drawn.
        If the demand can not be satisfied the agent cannot consum.
        If the demand is to low not consumed goods get wasted and do not increase firm`s profit.
        """
        agents = demand.keys()  # List of keys
        random.shuffle(agents)
        consumption = {}
        sold_goods = 0.0
        for agent in agents:
            agent_demand = demand[agent.agent_id]
            if agent_demand >= (self.production - sold_goods):
                d = agent_demand
            elif (self.production - sold_goods) >= 0.0:
                d = self.production - sold_goods
            else:
                raise ValueError
            consumption[agent.agent_id] = d
            sold_goods += d

        return consumption

    def earn_profits(self, consumption):
        self.profit = sum([consumption[agent.agent_id] * self.price for agent in consumption.keys()])
