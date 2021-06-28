from environment.base_env import BaseEnv

n_steps = 100000


def act(ob):
    return env.action_space.sample()


if __name__ == "__main__":
    env = BaseEnv(2)
    ob = env.reset()

    for step in range(n_steps):
        actions = {}
        for idx in range(env.n_agents):
            actions[idx] = act(ob)
        ob_next, r, done, _ = env.step(actions)
        if done:
            ob = env.reset()
        else:
            ob = ob_next
