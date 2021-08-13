import numpy as np
from gym import spaces
from human_friction.environment.simple_nk import SimpleNewKeynes
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

OBS_SPACE_AGENT = spaces.Dict(
    {
        "average_wage_increase": spaces.Box(0.0, 0.1, shape=(1,), dtype=np.float32),
        "average_consumption": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
        "budget": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        "inflation": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        "employed_hours": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        "interest": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
        "unemployment": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        "action_mask": spaces.Box(0.0, 1.0, shape=(50,), dtype=np.float32),
    }
)

# Actions of the format consumption x%, reservation wage x%
ACT_SPACE_AGENT = spaces.Discrete(50)


class RllibDiscrete(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = SimpleNewKeynes(env_config)
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
