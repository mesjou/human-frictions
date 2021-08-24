import pytest

import numpy as np
from human_friction.environment.simple_nk import SimpleNewKeynes
from human_friction.rllib.rllib_discrete import RllibDiscrete


def act(env):
    return {agent.agent_id: env.action_space.sample() for agent in env.wrapped_env.agents.values()}


def test_env():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    n_steps = 20
    rllib_env = RllibDiscrete(config)
    ob = rllib_env.reset()  # noqa E841
    for step in range(n_steps):
        action = act(rllib_env)
        ob_next, r, done, _ = rllib_env.step(action)
        if done["__all__"]:
            ob = rllib_env.reset()  # noqa E841
        else:
            ob = ob_next  # noqa E841
        assert isinstance(r, dict)
        assert isinstance(done, dict)
        assert isinstance(ob_next, dict)


def test_step():
    config = {"episode_length": 20, "n_agents": 2, "seed": 1}
    rllib_env = RllibDiscrete(config)
    nk_env = SimpleNewKeynes(config)

    ob_rllib = rllib_env.reset()
    ob_nk = nk_env.reset()
    for agent_id in ob_rllib.keys():
        assert (ob_nk[agent_id]["action_mask"] == ob_rllib[agent_id]["action_mask"]).all()
        del ob_nk[agent_id]["action_mask"]
        assert pytest.approx(sum(ob_nk[agent_id].values())) == ob_rllib[agent_id]["state"].sum()

    for step in range(20):
        action = act(rllib_env)
        ob_next_rllib, r_rllib, done_rllib, _ = rllib_env.step(action)
        ob_next_nk, r_nk, done_nk, _ = nk_env.step(action)

        for agent_id in ob_rllib.keys():
            assert (ob_next_nk[agent_id]["action_mask"] == ob_next_rllib[agent_id]["action_mask"]).all()
            del ob_next_nk[agent_id]["action_mask"]
            assert pytest.approx(sum(ob_next_nk[agent_id].values())) == ob_next_rllib[agent_id]["state"].sum()

        assert r_rllib == r_nk
        assert done_rllib == done_nk


def test_config():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 3.0, "labor_coefficient": 0.5}
    rllib_env = RllibDiscrete(config)
    ob = rllib_env.reset()  # noqa E841
    action = act(rllib_env)
    ob_next, r, done, _ = rllib_env.step(action)
    assert rllib_env.wrapped_env.labor_coefficient == 0.5
    assert rllib_env.wrapped_env.init_budget == 3.0


def test_flatten_observation():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 3.0, "labor_coefficient": 0.5}
    rllib_env = RllibDiscrete(config)
    input_dict = {
        "agent-0": {"a": 1.0, "b": 2.0, "c": 3.0, "action_mask": np.array([1.0, 0.0, 1.0])},
        "agent-1": {"b": 20.0, "a": 10.0, "c": 5.0},
    }
    output_dict = {
        "agent-0": {"state": np.array([1.0, 2.0, 3.0]), "action_mask": np.array([1.0, 0.0, 1.0])},
        "agent-1": {"state": np.array([10.0, 20.0, 5.0])},
    }

    for agent_id in output_dict.keys():
        assert compare_exact(rllib_env._flatten_observations(input_dict)[agent_id], output_dict[agent_id])


def compare_exact(first, second):
    """Return whether two dicts of arrays are exactly equal"""
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)
