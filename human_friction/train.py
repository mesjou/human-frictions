from human_friction.environment.single_env import LifeCycle

n_steps = 10


def act(ob):
    return env.action_space.sample()


config = {
    "episode_length": 10,
    "retirement_date": 7,
}

if __name__ == "__main__":
    env = LifeCycle(config)
    ob = env.reset()

    for step in range(n_steps):
        action = act(ob)
        ob_next, r, done, _ = env.step(action)
        print(ob_next)
        print(r)
        if done:
            ob = env.reset()
        else:
            ob = ob_next
