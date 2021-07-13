from human_friction.environment.single_env import LifeCycle


def test_env():
    def act(ob):
        if ob[0] <= 6:
            return [1.0]
        if 6 < ob[0] < 11:
            return [0.0]
        if 11 <= ob[0]:
            return [1.0]

        # return env.action_space.sample()

    config = {
        "episode_length": 20,
        "retirement_date": 15,
    }
    n_steps = 20

    env = LifeCycle(config)
    obs = env.reset()
    for step in range(n_steps):
        action = act(obs)
        ob_next, r, done, _ = env.step(action)
        if done:
            ob = env.reset()  # noqa F841
        else:
            ob = ob_next  # noqa F841
        assert isinstance(r, float)
        assert isinstance(done, bool)
        assert isinstance(ob_next, list)
