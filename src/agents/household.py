from agents.base_agent import Agent


class HouseholdAgent(Agent):
    def __init__(self, agent_id: str, budget: float):
        super().__init__(agent_id)

        assert isinstance(budget, float)
        self.budget: float = budget
        self.labor: float = 0.0
        self.consumption: float = 0.0

    def consume(self, fraction: float):
        """Defines how much an agent consumes and how the budget is affected"""
        consumption = fraction * self.budget
        assert consumption >= 0.0
        self.consumption = consumption
        self.budget -= consumption
