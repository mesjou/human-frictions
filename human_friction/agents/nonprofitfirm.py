from typing import Dict


class SimpleFirm(object):
    def __init__(
        self, technology: float = 0.5, alpha: float = 0.25,
    ):
        assert isinstance(technology, float)
        assert technology >= 0.0
        self.technology = technology

        assert isinstance(alpha, float)
        assert alpha >= 0.0
        self.alpha = alpha

        self.price: float = 1.0
        self.production: float = 0.0
        self.labor_costs: float = 0.0
        self.labor_demand: float = 0.01

    def produce(self, demand: Dict):
        production = sum([demand[agent_id] for agent_id in demand.keys()])
        assert production >= 0.0
        self.production = production

    def get_labor_demand(self):
        labor_demand = (self.production / self.technology) ** (1 / (1 - self.alpha))
        assert labor_demand >= 0.0
        self.labor_demand = labor_demand

    def hire_worker(self, wages: Dict) -> Dict:
        """Determine the hiring status based on reservation wages.

        The firm hires agents based on the wages. Lowest wages are hired first until labor demand is satisfied.
        When hired wage is payed to the agents and profit of firm is reduced.
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

    def set_price(self, occupation: Dict, wages: Dict) -> float:
        """
        The firm sets the price as a markup on the marginal costs.
        Args:
            occupation (dict): a dictionary that gives occupation per agent {agent_id: occupation between 0.0 and 1.0}.
            wages (dict): a dictionary that gives the agents' reservation wage {agent_id: wage}.

        Returns:
            inflation (float): the inflation as a change in the price due to the price setting by the firm.
        """
        labor_costs = sum([occupation[agent_id] * wages[agent_id] for agent_id in wages.keys()])
        try:
            new_price = labor_costs / self.production
        except ZeroDivisionError:
            new_price = 0.0

        inflation = (new_price - self.price) / self.price
        assert new_price >= 0.0, "Firms do not set price below zero"
        self.price = new_price

        return inflation
