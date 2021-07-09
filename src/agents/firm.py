import random
from typing import Dict


class Firm(object):
    def __init__(
        self,
        init_labor_demand: float,
        technology: float = 0.5,
        alpha: float = 0.25,
        learning_rate: float = 0.01,
        markup: float = 0.1,
        memory: float = 0.45,
    ):
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

        assert isinstance(memory, float)
        assert 1.0 > memory >= 0.0
        self.memory = memory

        assert isinstance(init_labor_demand, float)
        assert init_labor_demand >= 0.0
        self.labor_demand: float = init_labor_demand

        self.average_profit: float = 0.0
        self.profit: float = 0.0
        self.price: float = 0.0
        self.production: float = 0.0
        self.labor_costs: float = 0.0

    def produce(self, occupation: Dict):
        labor = sum([occupation[agent_id] for agent_id in occupation.keys()])
        assert labor >= 0.0
        self.production = self.technology * labor ** (1 - self.alpha)

    def learn(self, max_labor):
        if self.profit >= self.average_profit:
            labor_demand = self.labor_demand * (1 + self.learning_rate)
        else:
            labor_demand = self.labor_demand * (1 - self.learning_rate)

        # assert that labor demand is never zero
        self.labor_demand = max(min(labor_demand, max_labor), self.learning_rate)
        self.average_profit = (1 - self.memory) * self.profit + self.memory * self.average_profit

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
        self.labor_costs = sum([occupation[agent_id] * wages[agent_id] for agent_id in wages.keys()])
        new_price = (1 + self.markup) / (1 - self.alpha) * self.labor_costs / self.production
        inflation = (new_price - self.price) / self.price
        self.price = new_price
        return inflation

    def sell_goods(self, demand: Dict):
        """Determine how much the firm sells to which agent.

        The order of the selling process is randomly drawn.
        If the demand can not be satisfied the agent cannot consum.
        If the demand is to low not consumed goods get wasted and do not increase firm`s profit.
        """
        agent_ids = list(demand.keys())  # List of keys
        random.shuffle(agent_ids)
        consumption = {}
        sold_goods = 0.0
        for agent_id in agent_ids:
            agent_demand = demand[agent_id]
            if agent_demand >= (self.production - sold_goods):
                d = agent_demand
            elif (self.production - sold_goods) >= 0.0:
                d = self.production - sold_goods
            else:
                raise ValueError
            consumption[agent_id] = d
            sold_goods += d

        return consumption

    def earn_profits(self, consumption):
        self.profit = sum([consumption[agent_id] * self.price for agent_id in consumption.keys()])
