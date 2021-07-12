import pytest

from environment.new_keynes import NewKeynesMarket


def test_env():
    config = {"episode_length": 20, "n_agents": 4, "init_budget": 0.0}

    def act(env: NewKeynesMarket):
        return {agent.agent_id: env.action_space.sample() for agent in env.agents.values()}

    n_steps = 20
    env = NewKeynesMarket(config)
    ob = env.reset()  # noqa E841
    for step in range(n_steps):
        action = act(env)
        ob_next, r, done, _ = env.step(action)
        if done:
            ob = env.reset()  # noqa E841
        else:
            ob = ob_next  # noqa E841
        # print(ob, action, r)
        assert isinstance(r, dict)
        assert isinstance(done, dict)
        assert isinstance(ob_next, dict)


def test_clear_labor_market():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    wages = {"agent-0": 0.5627646, "agent-1": 0.0175226}
    env = NewKeynesMarket(config)
    env.reset()
    env.clear_labor_market(wages)

    # initial labor demand is 1 for each agent
    assert env.agents["agent-0"].budget > env.agents["agent-1"].budget
    for agent in env.agents.values():
        assert agent.labor == 1.0
    assert env.firm.production > 0.0
    assert env.inflation > 0.0
    assert env.unemployment == 0.0


def test_get_unemployment():
    config = {"episode_length": 20, "n_agents": 4, "init_budget": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    assert env.get_unemployment() == 0.0

    env.firm.labor_demand = 2.0
    assert env.get_unemployment() == 0.5

    with pytest.raises(Exception):
        env.firm.labor_demand = 0.0
        assert env.get_unemployment() == 1.0

    with pytest.raises(Exception):
        env.firm.labor_demand = -1.0
        env.get_unemployment()


def test_clear_goods_market():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.firm.price = 1.0
    demand = {"agent-0": 0.5, "agent-1": 0.3}

    # nothing to sell
    env.firm.production = 0.0
    env.firm.average_profit = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget == env.agents["agent-1"].budget
    assert env.agents["agent-0"].budget == 0.0
    assert env.firm.profit == 0.0
    assert env.firm.labor_demand == 2 * 0.99

    # production of 1 to sell
    env.firm.production = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget < env.agents["agent-1"].budget
    assert env.agents["agent-0"].budget < 0.0
    assert env.agents["agent-1"].budget < 0.0
    assert env.agents["agent-0"].consumption > env.agents["agent-1"].consumption
    assert env.firm.profit == 0.8
    assert env.firm.labor_demand == 2 * 0.99 * 1.01


if __name__ == "__main__":
    test_clear_labor_market()
    test_get_unemployment()
    test_env()
