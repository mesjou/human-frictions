import random

import gym
import numpy as np
from gym.spaces import Box


class LifeCycle(gym.Env):
    def __init__(self, config):
        self.t = 0.0
        self.budget = 0.0
        self.income = 1.0
        self.retirement = 0.8
        self.T = config["episode_length"]
        self.retirement_date = config["retirement_date"]
        self.action_space = Box(0.0, np.inf, shape=(1,))
        self.observation_space = Box(
            low=np.array([0.0, -np.inf, 0.0]),
            high=np.array([config["episode_length"], np.inf, np.inf]),
            dtype=np.float32,
        )

        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(None)

    def reset(self):
        self.t = 0
        self.budget = 0
        return [self.t, self.budget, self.income]

    def step(self, action):
        self.t += 1
        cash_inflow = self.income if self.t <= self.retirement_date else self.retirement
        self.budget += -action[0] + cash_inflow

        if self.t == self.T and self.budget < 0:
            reward = -60.0
        else:
            reward = np.log(action[0] + 1.0)
        next_cash_flow = self.income if self.t + 1 <= self.retirement_date else self.retirement
        obs = [self.t, self.budget, next_cash_flow]
        done = False if self.t < self.T else True
        return obs, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)
