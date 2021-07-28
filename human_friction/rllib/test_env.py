import numpy as np
from gym import spaces
from human_friction.environment.new_keynes import NewKeynesMarket
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from pprint import pprint

OBS_SPACE_AGENT = spaces.Dict(
    {
        "average_wage": spaces.Box(0.0, np.inf, shape=(1,)),
        "budget": spaces.Box(-np.inf, np.inf, shape=(1,)),
        "inflation": spaces.Box(-np.inf, np.inf, shape=(1,)),
        "interest": spaces.Box(-np.inf, np.inf, shape=(1,)),
        "unemployment": spaces.Box(0.0, 1.0, shape=(1,)),
    }
)

env_config = {
    "episode_length": 800,
    "n_agents": 4,
}


env = NewKeynesMarket(env_config)
env.reset()
actions = {
            id:[0,i]
            for id,i in zip(env.agents.keys(),range(env_config["n_agents"]))
            }
print("Test actions:")
pprint(actions)
wages, demand = env.parse_actions(actions)
pprint("Wages: {}".format(wages))
pprint("Demand: {}".format(demand))

print("Test labor market")
env.clear_labor_market(wages)

pprint("Firm production {}".format(env.firm.production))
pprint("New price {}".format(env.firm.price))
pprint("Inflation {}".format(env.inflation))
pprint("Unemployment {}".format(env.unemployment))

pprint(env.step(actions))
