from human_friction.environment.new_keynes import NewKeynesMarket
from human_friction.rllib.rllib_env import RllibEnv


def act(env):
    return {agent.agent_id: env.action_space.sample() for agent in env.env.agents.values()}


def test_env():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 0.0}
    n_steps = 20
    rllib_env = RllibEnv(config)
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
    rllib_env = RllibEnv(config)
    nk_env = NewKeynesMarket(config)

    ob_rllib = rllib_env.reset()
    ob_nk = nk_env.reset()
    assert ob_rllib == ob_nk

    for step in range(20):
        action = act(rllib_env)
        ob_next_rllib, r_rllib, done_rllib, _ = rllib_env.step(action)
        ob_next_nk, r_nk, done_nk, _ = nk_env.step(action)

        assert ob_next_rllib == ob_next_nk
        assert r_rllib == r_nk
        assert done_rllib == done_nk


def test_config():
    config = {"episode_length": 20, "n_agents": 2, "init_budget": 3.0, "labor_coefficient": 0.5}
    rllib_env = RllibEnv(config)
    ob = rllib_env.reset()  # noqa E841
    action = act(rllib_env)
    ob_next, r, done, _ = rllib_env.step(action)
    assert rllib_env.env.labor_coefficient == 0.5
    assert rllib_env.env.init_budget == 3.0
