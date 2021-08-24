import numpy as np
from gym import spaces
from human_friction.environment.simple_nk import SimpleNewKeynes
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

OBS_SPACE_AGENT = spaces.Dict(
    {
        "action_mask": spaces.Box(0.0, 1.0, shape=(50,)),
        "state": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
    }
)
ACT_SPACE_AGENT = spaces.Discrete(50)


class RllibDiscrete(MultiAgentEnv):
    def __init__(self, env_config):
        self.wrapped_env = SimpleNewKeynes(env_config)
        self.observation_space = OBS_SPACE_AGENT
        self.action_space = ACT_SPACE_AGENT

    def reset(self) -> MultiAgentDict:
        wrapped_obs = self.wrapped_env.reset()
        obs = self._flatten_observations(wrapped_obs)

        return obs

    def step(self, actions: MultiAgentDict) -> (MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict):
        wrapped_obs, r, done, info = self.wrapped_env.step(actions)
        obs = self._flatten_observations(wrapped_obs)

        return obs, r, done, info

    @staticmethod
    def _flatten_observations(obs):
        """
        Take observations of the form {agent_id: {key1: value1, key2: value2 ... action_mask: [0, 1, 2]}}
        and transforms it to {agent_id: {state: np.array(value1, value2 ...], action_mask: [0, 1, 2]}}.

        Args:
            obs (dict): original observation.

        Returns:
            flatten_obs (dict): observation that separates action mask from other values.

        """
        flatten_obs = dict()
        for agent_id, observations in obs.items():
            flatten_obs[agent_id] = dict()
            if "action_mask" in observations.keys():
                flatten_obs[agent_id]["action_mask"] = observations["action_mask"]
                del observations["action_mask"]
            # be sure that the observation keys are always at the same position in the numpy array
            flatten_obs[agent_id]["state"] = np.fromiter(dict(sorted(observations.items())).values(), dtype=np.float32)

        return flatten_obs
