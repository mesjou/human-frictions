import pytest

from human_friction.environment.new_keynes import NewKeynesMarket


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


def test_clear_dividends():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.clear_dividends(10.0)
    for agent in env.agents.values():
        assert agent.budget == 5.0

    env.clear_dividends(-10.0)
    for agent in env.agents.values():
        assert agent.budget == 0.0

    env.clear_dividends(0.0)
    for agent in env.agents.values():
        assert agent.budget == 0.0

    config = {"episode_length": 20, "n_agents": 10, "init_budget": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.clear_dividends(10.0)
    for agent in env.agents.values():
        assert agent.budget == 1.0


def test_clear_capital_market():

    # target is reached and interest = 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.inflation = 0.02
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget == 2.04

    # inflation too high thus interest > 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.inflation = 0.03
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget > 2.04

    # inflation too low: interest < 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.inflation = 0.01
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget < 2.04

    # unemployment too high thus interest < 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.inflation = 0.02
    env.unemployment = 0.5
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget < 2.04

    # unemployment too low: interest > 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.central_bank.natural_unemployment = 0.2
    env.inflation = 0.02
    env.unemployment = 0.1
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget > 2.04
