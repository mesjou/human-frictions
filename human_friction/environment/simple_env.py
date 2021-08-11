from typing import Dict, Tuple
from human_friction.environment.base_env import BaseEnv
from human_friction.utils import rewards
from human_friction.agents.simple_hh import SimpleAgent
from ray.rllib.utils.typing import MultiAgentDict
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv



class SimpleEnv(BaseEnv):

    def __init__(self, config):

        super().__init__(config)

        # Optional parameters
        # ----------
        wage = config.get("wage", 10.0)
        assert isinstance(wage, float)
        assert wage > 0.0
        self.wage = wage

        init_budget = config.get("init_budget", 0.0)
        assert isinstance(init_budget, float)
        assert wage >= 0.0
        self.init_budget = init_budget

        retirement_age =  config.get("retirement_age", 4)
        assert isinstance(retirement_age, int)
        assert wage >= 0.0
        self.retirement_age = retirement_age

        init_interest =  config.get("interest", 0.02)
        assert isinstance(init_interest, float)
        assert init_interest >= 0.0
        self.init_interest = init_interest

        # Final setup of the environment
        # ----------
        self.agents: Dict[str:SimpleAgent] = {}
        self.action_space = spaces.Discrete(int(self.wage))

        self.observation_space = spaces.Dict(
            {
                "next_period_wage": spaces.Box(0.0, np.inf, shape=(1,)),
                "budget": spaces.Box(0.0, np.inf, shape=(1,)),
                "interest": spaces.Box(0.0, np.inf, shape=(1,))
            }
        )
        self.INTEREST_SCHEME = [self.init_interest]* (config["episode_length"]+1)
        self.INTEREST_SCHEME[2] = 50*self.init_interest

    def set_up_agents(self):
        """Initialize the agents and give them starting endowment."""
        for idx in range(self.n_agents):
            agent_id = "agent-" + str(idx)
            agent = SimpleAgent(agent_id, self.init_budget)
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
                "next_period_wage": self.wage,
                "budget": self.init_budget,
                "interest": self.init_interest
            }

        return obs

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.timestep += 1

        obs = self.generate_observations(actions)
        rew = self.compute_rewards()
        done = {"__all__": self.timestep >= self._episode_length}
        info = self.generate_info()

        return obs, rew, done, info

    def generate_observations(self, actions: MultiAgentDict) -> MultiAgentDict:
        self.interest = self.INTEREST_SCHEME[self.timestep]

        consumption = self.parse_actions(actions)

        paid_wage = self.wage if self.timestep < self.retirement_age else 0
        next_period_wage = self.wage if self.timestep+1< self.retirement_age else 0

        self.earn(paid_wage)
        self.map_actions(consumption)
        self.update_budget()

        obs = {}
        for agent in self.agents.values():
            obs[agent.agent_id] = {
                "next_period_wage": next_period_wage,
                "budget": agent.budget,
                "interest": self.interest
            }

        return obs

    def map_actions(self, consumption):
        for agent in self.agents.values():
            c = max(min(consumption[agent.agent_id], agent.budget), 0.0)
            agent.consumption = c


    def compute_rewards(self) -> MultiAgentDict:
        """Consumptions >=0 and consumption <= budget
        """
        rew = {}
        for agent in self.agents.values():
            rew[agent.agent_id] = agent.consumption#np.log(1+agent.consumption)
        return rew

    def generate_info(self):
        info = {}
        for agent in self.agents.values():
            info[agent.agent_id] = {
                'interets':self.interest,
                'end_of_period_budget':agent.budget,
                'actual_consumption': agent.consumption,
            }
        return info

    def parse_actions(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict]:
        consumption = {}
        for agent in self.agents.values():
            agent_action = actions[agent.agent_id]
            consumption[agent.agent_id] = agent_action+0.01
        return consumption

    def earn(self, wage: float) -> MultiAgentDict:
        for agent in self.agents.values():
            agent.budget *=  1 + self.interest
            agent.budget += wage


    def update_budget(self) -> MultiAgentDict:
        for agent in self.agents.values():
            agent.budget -= agent.consumption
