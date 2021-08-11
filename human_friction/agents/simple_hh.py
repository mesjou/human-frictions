from human_friction.agents.base_agent import Agent


class SimpleAgent(Agent):
    def __init__(self, agent_id: str, budget: float):
        super().__init__(agent_id)

        assert isinstance(budget, float)
        self.budget: float = budget
        self.consumption: float = 0.0
