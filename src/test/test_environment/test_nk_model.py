from environment.salle_like import NewKeynesMarket


def test_env():
    def act(env: NewKeynesMarket):
        return {agent.agent_id: env.action_space.sample() for agent in env.agents.values()}

    config = {"episode_length": 20, "n_agents": 4, "init_budget": 0.0}
    n_steps = 20

    env = NewKeynesMarket(config)
    ob = env.reset()  # noqa E841
    for step in range(n_steps):
        action = act(env)
        ob_next, r, done, _ = env.step(action)
        if done:
            ob = env.reset()
        else:
            ob = ob_next
        print(ob, action, r)
        assert isinstance(r, dict)
        assert isinstance(done, dict)
        assert isinstance(ob_next, dict)


if __name__ == "__main__":
    test_env()
