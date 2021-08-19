import ray
from human_friction.rllib.models import FCNet
from human_friction.rllib.train_action_masking import rllib_config
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", FCNet)


def run():
    config = {
        **rllib_config,
        "num_workers": 0,
        "num_envs_per_worker": 1,
    }
    trainer = PPOTrainer(config=config)
    while True:
        print(trainer.train())


if __name__ == "__main__":
    ray.init()
    run()
    ray.shutdown()
