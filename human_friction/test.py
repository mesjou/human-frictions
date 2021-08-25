import ray
from human_friction.environment.simple_nk import SimpleNewKeynes
from human_friction.rllib.models import FCNet
from human_friction.run_configurations.environment_config import env_config
from human_friction.run_configurations.rllib_config import rllib_config
from human_friction.utils.metric_generator import MetricGenerator
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", FCNet)


def load(path, config):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    agent = PPOTrainer(config=config, env=config["env"])
    agent.restore(path)
    return agent


def test(agent, config):
    """Test trained agent for a single episode."""
    # instantiate env class
    env_configuration = config.get("env_config", env_config)
    environment = config.get("env", SimpleNewKeynes)
    env = environment(env_configuration)

    # run until episode ends
    done = False
    obs = env.reset()
    metric_generator = MetricGenerator(env)
    while not done:
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = config["multiagent"]["policy_mapping_fn"](agent_id)
            action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id)
        obs, reward, done, info = env.step(action)
        metric_generator.analyze(env, reward)
        done = done["__all__"]

    metric_generator.plot()


if __name__ == "__main__":
    ray.init()
    ppo_agent = load(
        "run_configurations/checkpoints/PPO_New_Keynes/"
        + "PPO_RllibDiscrete_2ed37_00000_0_2021-08-24_15-07-06/checkpoint_002000/checkpoint-2000",  # noqa W503
        rllib_config,
    )
    test(ppo_agent, rllib_config)
    ray.shutdown()
