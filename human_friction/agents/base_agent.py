class Agent(object):
    def __init__(self, agent_id: str):
        """
        Superclass for all agents.

        Args
            agent_id (str): a unique id allowing the map to identify the agents
        """
        assert isinstance(agent_id, str)
        self.agent_id = agent_id
