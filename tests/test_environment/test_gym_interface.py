from human_friction.environment.new_keynes import NewKeynesMarket


def act(env: NewKeynesMarket):
    return {agent.agent_id: env.action_space.sample() for agent in env.agents.values()}


def test_env():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    n_steps = 20
    env = NewKeynesMarket(config)
    ob = env.reset()  # noqa E841
    for step in range(n_steps):
        action = act(env)
        ob_next, r, done, _ = env.step(action)
        if done["__all__"]:
            ob = env.reset()  # noqa E841
        else:
            ob = ob_next  # noqa E841
        assert isinstance(r, dict)
        assert isinstance(done, dict)
        assert isinstance(ob_next, dict)


def test_compute_rewards():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    for agent in env.agents.values():
        agent.labor = 1.0
        agent.consumption = 1.0
    rew = env.compute_rewards()
    for agent_id in env.agents.keys():
        assert rew[agent_id] < 0.0

    for agent in env.agents.values():
        agent.labor = 0.0
        agent.consumption = 0.0
    rew = env.compute_rewards()
    for agent_id in env.agents.keys():
        assert rew[agent_id] == 0.0

    i = 1.0
    for agent in env.agents.values():
        agent.labor = i
        agent.consumption = 1.0
        i += 1.0
    rew = env.compute_rewards()
    agent_ids = list(env.agents.keys())
    assert rew[agent_ids[0]] > rew[agent_ids[1]]
