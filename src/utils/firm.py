from typing import Dict


class Firm(object):
    def __init__(self, technology: float = 0.5, alpha: float = 0.2, learning_rate: float = 0.1):
        assert isinstance(technology, float)
        assert technology >= 0.0
        self.technology = technology

        assert isinstance(alpha, float)
        assert alpha >= 0.0
        self.alpha = alpha

        assert isinstance(learning_rate, float)
        assert learning_rate >= 0.0
        self.learning_rate = learning_rate

        self.labor_demand: float = 0.0
        self.average_profit: float = 0.0
        self.profit: float = 0.0
        self.price: float = 0.0
        self.profit: float = 0.0

    def produce(self, labor_demand: float):
        assert labor_demand >= 0.0
        return self.technology * labor_demand ** (1 - self.alpha)

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
