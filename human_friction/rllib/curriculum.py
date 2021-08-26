from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext


def curriculum_fn(train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.
    Args:
        train_results (dict): The train results returned by Trainer.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.
    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    if train_results["episode_reward_mean"] > 860:
        new_task = 4.0
    elif train_results["episode_reward_mean"] > 830:
        new_task = 6.0
    elif train_results["episode_reward_mean"] > 800:
        new_task = 8.0
    elif train_results["episode_reward_mean"] > 700:
        new_task = 10.0
    else:
        new_task = 20.0

    return new_task
