from typing import Sequence

import pandas as pd
from human_friction.rllib.rllib_env import RllibEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker


class MyCallbacks(DefaultCallbacks):
    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        envs: Sequence[RllibEnv] = base_env.get_unwrapped()
        social_metrics = pd.DataFrame([e.env.scenario_metrics() for e in envs]).mean().to_dict()

        for k, v in social_metrics.items():
            episode.custom_metrics[k] = v
        # print(episode.hist_stats)
