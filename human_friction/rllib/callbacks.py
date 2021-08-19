from typing import Sequence

import numpy as np
import pandas as pd
from human_friction.rllib.rllib_env import RllibEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        envs: Sequence[RllibEnv] = base_env.get_unwrapped()
        social_metrics = pd.DataFrame([e.wrapped_env.get_custom_metrics() for e in envs]).mean().to_dict()
        for k, _ in social_metrics.items():
            episode.user_data[k] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        envs: Sequence[RllibEnv] = base_env.get_unwrapped()
        social_metrics = pd.DataFrame([e.wrapped_env.get_custom_metrics() for e in envs]).mean().to_dict()
        for k, v in social_metrics.items():
            episode.user_data[k].append(v)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        for k, v in episode.user_data.items():
            average_value = np.mean(episode.user_data[k])
            episode.custom_metrics[k] = average_value
