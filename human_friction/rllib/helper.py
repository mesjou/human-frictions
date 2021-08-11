import numpy as np
from gym import spaces
from human_friction.environment.simple_env import SimpleEnv
from human_friction.environment.new_keynes import NewKeynesMarket
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

OBS_SPACE_AGENT = spaces.Dict(
    {
        "next_period_wage": spaces.Box(0.0, np.inf, shape=(1,)),
        "budget": spaces.Box(0.0, np.inf, shape=(1,)),
        "interest": spaces.Box(0.0, np.inf, shape=(1,)),
    }
)
ACT_SPACE_AGENT =  spaces.Discrete(10)
#ACT_SPACE_AGENT = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)

class SimpleRllibEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = SimpleEnv(env_config)
        self.observation_space = OBS_SPACE_AGENT
        self.action_space = ACT_SPACE_AGENT

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        return obs

    def step(self, actions: MultiAgentDict) -> (MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict):
        obs, r, done, info = self.env.step(actions)
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        return obs, r, done, info
