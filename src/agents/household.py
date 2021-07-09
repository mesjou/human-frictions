from agents.base_agent import Agent


class HouseholdAgent(Agent):
    def __init__(self, agent_id: str, budget: float):
        super().__init__(agent_id)

        assert isinstance(budget, float)
        self.budget: float = budget
        self.labor: float = 0.0
        self.consumption: float = 0.0

    def consume(self, consumption: float, price: float):
        """Defines how much an agent consumes and how the budget is affected"""
        assert consumption >= 0.0
        assert price >= 0.0
        self.consumption = consumption
        self.budget -= consumption * price

    def earn(self, hours_worked, wage):
        assert hours_worked >= 0.0
        assert wage >= 0.0
        self.labor = hours_worked
        self.budget += hours_worked * wage
