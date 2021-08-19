import pytest

import numpy as np
from human_friction.environment.simple_nk import SimpleNewKeynes


def test_clear_labor_market():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    demand = {"agent-0": 0.01, "agent-1": 0.01}
    wages = {"agent-0": 0.5627646, "agent-1": 0.0175226}
    env = SimpleNewKeynes(config)
    env.reset()
    env.clear_markets(demand, wages)

    # initial labor demand is 1 for each agent
    assert env.agents["agent-0"].budget <= env.agents["agent-1"].budget
    assert env.agents["agent-0"].labor <= env.agents["agent-1"].labor
    assert env.firm.production == 0.02
    assert env.inflation != 0.0
    assert env.unemployment > 0.0


def test_get_unemployment():
    config = {"episode_length": 20, "n_agents": 4, "init_budget": 0.0}
    env = SimpleNewKeynes(config)
    env.reset()
    assert env.get_unemployment() == (4 - 0.01) / 4

    env.firm.labor_demand = 2.0
    assert env.get_unemployment() == 0.5

    with pytest.raises(Exception):
        env.firm.labor_demand = 0.0
        assert env.get_unemployment() == 1.0

    with pytest.raises(Exception):
        env.firm.labor_demand = -1.0
        env.get_unemployment()


def test_clear_capital_market():

    # target is reached and interest = 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = SimpleNewKeynes(config)
    env.reset()
    env.inflation = 0.02
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget == 2.04

    # inflation too high thus interest > 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = SimpleNewKeynes(config)
    env.reset()
    env.inflation = 0.03
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget > 2.04

    # inflation too low: interest < 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = SimpleNewKeynes(config)
    env.reset()
    env.inflation = 0.01
    env.unemployment = 0.0
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget < 2.04

    # unemployment too high thus interest < 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = SimpleNewKeynes(config)
    env.reset()
    env.inflation = 0.02
    env.unemployment = 0.5
    env.clear_capital_market()
    for agent in env.agents.values():
        assert agent.budget < 2.04

    # unemployment too low: interest > 1.02
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 2.0}
    env = SimpleNewKeynes(config)
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
    env = SimpleNewKeynes(config)
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
    env = SimpleNewKeynes(config)
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
    config = {"episode_length": 20, "n_agents": 3, "alpha": 0.0}
    env = SimpleNewKeynes(config)
    env.reset()
    actions = {"agent-0": 0, "agent-1": 2, "agent-2": 1}
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert pytest.approx(agent_obs["average_wage_increase"]) == 0.025
        assert agent_obs["average_consumption"] == 0.01

    actions = {"agent-0": 40, "agent-1": 42, "agent-2": 41}
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert pytest.approx(agent_obs["average_wage_increase"]) == 0.025
        assert agent_obs["average_consumption"] == 0.01 + (1.0 - 0.01) / 9 * 8

    actions = {"agent-0": 40, "agent-1": 45, "agent-2": 20}
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert agent_obs["average_wage_increase"] == 0.0
        assert agent_obs["average_consumption"] == np.mean([0.01 + (1 - 0.01) / 9 * 8, 1.0, 0.01 + (1 - 0.01) / 9 * 4])

    actions = {"agent-0": 49, "agent-1": 44, "agent-2": 4}
    obs = env.generate_observations(actions)
    for agent_id, agent_obs in obs.items():
        assert pytest.approx(agent_obs["average_wage_increase"]) == 0.1
        assert pytest.approx(agent_obs["average_consumption"]) == np.mean([0.01, 0.01 + (1 - 0.01) / 9 * 8, 1])

    # TODO think if we should base observation on average wage increase that was realized
    # env.firm.labor_demand = 2.0
    # obs = env.generate_observations(actions)
    # for agent_id, agent_obs in obs.items():
    #    assert agent_obs["average_wage_increase"] == 0.2
    #
    # env.firm.labor_demand = 1.5
    # obs = env.generate_observations(actions)
    # for agent_id, agent_obs in obs.items():
    #    assert agent_obs["average_wage_increase"] == (0.5 * 0.3 + 1.0 * 0.1) / 1.5


def test_scale():
    config = {"episode_length": 20, "n_agents": 3}
    env = SimpleNewKeynes(config)
    env.reset()

    n_w_actions = 5
    n_c_actions = 10
    for i in np.arange(0.01, env.get_max_consumption(), 42):
        assert env.scale(i, n_w_actions=n_w_actions, n_c_actions=n_c_actions) % n_w_actions == 0

    n_w_actions = 6
    n_c_actions = 9
    for i in np.arange(0.01, env.get_max_consumption(), 42):
        assert env.scale(i, n_w_actions=n_w_actions, n_c_actions=n_c_actions) % n_w_actions == 0

    n_w_actions = 2
    n_c_actions = 6
    for i in np.arange(0.01, env.get_max_consumption(), 42):
        assert env.scale(i, n_w_actions=n_w_actions, n_c_actions=n_c_actions) % n_w_actions == 0


def test_get_action_mask():
    config = {"episode_length": 20, "n_agents": 3, "init_budget": 0.0}
    env = SimpleNewKeynes(config)
    env.reset()
    assert sum(env.get_action_mask(env.agents["agent-0"])) == 5

    config = {"episode_length": 20, "n_agents": 3, "init_budget": 100.0}
    env = SimpleNewKeynes(config)
    env.reset()
    assert sum(env.get_action_mask(env.agents["agent-0"])) == 50

    config = {
        "episode_length": 20,
        "n_agents": 2,
        "init_budget": 1.0,
        "init_price": 1.0,
        "technology": 1.0,
        "alpha": 0.0,
    }
    env = SimpleNewKeynes(config)
    env.reset()
    assert sum(env.get_action_mask(env.agents["agent-0"])) == 50

    config = {
        "episode_length": 20,
        "n_agents": 2,
        "init_budget": 0.5,
        "init_price": 1.0,
        "technology": 1.0,
        "alpha": 0.0,
    }
    env = SimpleNewKeynes(config)
    env.reset()
    assert sum(env.get_action_mask(env.agents["agent-0"])) == 20

    config = {
        "episode_length": 20,
        "n_agents": 2,
        "init_budget": 0.25,
        "init_price": 1.0,
        "technology": 1.0,
        "alpha": 0.0,
    }
    env = SimpleNewKeynes(config)
    env.reset()
    assert sum(env.get_action_mask(env.agents["agent-0"])) == 10
