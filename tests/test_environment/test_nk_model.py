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
    assert env.inflation != 0.0
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
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 20.0}
    env = NewKeynesMarket(config)
    env.reset()
    env.firm.price = 1.0
    demand = {"agent-0": 0.5, "agent-1": 0.3}

    # nothing to sell
    env.firm.production = 0.0
    env.firm.average_profit = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget == env.agents["agent-1"].budget
    assert env.agents["agent-0"].budget == 20.0
    assert env.firm.profit == 0.0
    assert env.firm.labor_demand == 2 * 0.99

    # production of 1 to sell
    env.firm.production = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget < env.agents["agent-1"].budget
    assert env.agents["agent-0"].budget < 20.0
    assert env.agents["agent-1"].budget < 20.0
    assert env.agents["agent-0"].consumption > env.agents["agent-1"].consumption
    assert env.firm.profit == 0.8
    assert env.firm.labor_demand == 2 * 0.99 * 1.01

    # agent-0 has no budget
    env.agents["agent-0"].budget = 0.0
    env.firm.production = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget == 0.0
    assert env.agents["agent-1"].budget < 20.0
    assert env.agents["agent-1"].budget > 0.0
    assert env.agents["agent-0"].consumption == 0.0
    assert env.agents["agent-1"].consumption == 0.3
    assert env.firm.profit == 0.3

    # agent-0 has not enough budget
    env.agents["agent-0"].budget = 0.22
    env.firm.price = 0.5
    env.firm.production = 1.0
    env.clear_goods_market(demand)
    assert env.agents["agent-0"].budget == 0.22
    assert env.agents["agent-1"].budget < 20.0
    assert env.agents["agent-1"].budget > 0.0
    assert env.agents["agent-0"].consumption == 0.0
    assert env.agents["agent-1"].consumption == 0.3
    assert env.firm.profit == 0.15


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


def test_compute_rewards():

    # labor comes with disutility
    config = {"episode_length": 20, "n_agents": 2, "labor_coefficient": 1.0}
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

    # labor is for free, no utility loss
    config = {"episode_length": 20, "n_agents": 2, "labor_weight": 0.0}
    env = NewKeynesMarket(config)
    env.reset()
    for agent in env.agents.values():
        agent.labor = 1.0
        agent.consumption = 1.0
    rew = env.compute_rewards()
    for agent_id in env.agents.keys():
        assert rew[agent_id] > 0.0

    i = 1.0
    for agent in env.agents.values():
        agent.labor = i
        agent.consumption = 1.0
        i += 1.0
    rew = env.compute_rewards()
    agent_ids = list(env.agents.keys())
    assert rew[agent_ids[0]] == rew[agent_ids[1]]


def test_generate_observations():
    config = {"episode_length": 20, "n_agents": 3}
    env = NewKeynesMarket(config)
    env.reset()
    actions = {"agent-0": [0.0, 0.5], "agent-1": [0.0, 0.3], "agent-2": [0.0, 0.1]}
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert agent_obs["average_wage"] == 0.3

    env.firm.labor_demand = 2.0
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert agent_obs["average_wage"] == 0.2

    env.firm.labor_demand = 1.5
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert agent_obs["average_wage"] == (0.5 * 0.3 + 1.0 * 0.1) / 1.5
