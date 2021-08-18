from human_friction.rllib.rllib_discrete import RllibDiscrete


def act(env):
    return {agent.agent_id: env.action_space.sample() for agent in env.env.agents.values()}


def act_max(env):
    return {agent.agent_id: 49 for agent in env.env.agents.values()}


def act_min(env):
    return {agent.agent_id: 0 for agent in env.env.agents.values()}


def test_step():
    config = {"episode_length": 20, "n_agents": 10}
    rllib_env = RllibDiscrete(config)
    rllib_env.reset()
    for step in range(20):
        action = act(rllib_env)
        ob_next_rllib, r_rllib, done_rllib, _ = rllib_env.step(action)

    rllib_env.reset()
    for step in range(20):
        action = act_max(rllib_env)
        ob_next_rllib, r_rllib, done_rllib, _ = rllib_env.step(action)

    rllib_env.reset()
    for step in range(20):
        action = act_min(rllib_env)
        ob_next_rllib, r_rllib, done_rllib, _ = rllib_env.step(action)
