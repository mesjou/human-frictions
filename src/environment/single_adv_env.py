#http://gym.openai.com/docs/
#last modified on 06.07.21 by Anna Almosova
import random

import gym
import numpy as np
from gym.spaces import Box


class LifeCycle(gym.Env):
    def __init__(self, config):
        self.t = 0.0
        self.budget = 0.0
        self.investment = 0.0
        self.income = 1.0
        self.retirement = 0.8
        self.interest = 0.02
        #self.investment_return = 0.03
        self.T = config["episode_length"]
        self.retirement_date = config["retirement_date"]
        self.action_space = Box(low=np.array([0.0, 0.0]), high = np.array([np.inf, np.inf]), shape=(2,))
        self.observation_space = Box(
            low=np.array([0.0, -np.inf, 0.0, 0.0, -1.0]),
            high=np.array([config["episode_length"], np.inf, np.inf, np.inf, 1.0]),
            dtype=np.float32,
        )
        #observations are t, budget, investments, next income, interest, investment return
        #why next income?
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(None)

        #reset all observations?
    def reset(self):
        # reterns all reset observations
        self.t = 0
        self.budget = 0
        self.investment = 0
        self.interest = 0.02
        return [self.t, self.budget, self.investment, self.income, self.interest]

    def step(self, action, shock=None):
        #always returns observations (obj), reward (float), done (Bool), info (dict)
        self.t += 1
        cash_inflow = self.income if self.t <= self.retirement_date else self.retirement
        cash_inflow += (0.03-self.interest)*self.investment
        self.budget += -action[0] - action[1] + cash_inflow
        self.investment *= 0.9 #depreciation, too high
        self.investment += action[1]
        self.interest = 0.02*(1+0.2*(self.investment-10)) #MP rule

        if self.t == self.T and self.budget < 0:
            reward = -60
        else:
            reward = np.log(action[0] + 1.0)
        next_cash_flow = self.income if self.t + 1 <= self.retirement_date else self.retirement
        next_cash_flow += (0.03-self.interest) * self.investment
        obs = [self.t, self.budget, self.investment, next_cash_flow, self.interest]
        done = False if self.t < self.T else True
        return obs, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)
