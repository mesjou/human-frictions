import os

from human_friction.rllib.callbacks import MyCallbacks
from human_friction.rllib.rllib_discrete import ACT_SPACE_AGENT, OBS_SPACE_AGENT, RllibDiscrete
from human_friction.run_configurations.environment_config import env_config

rllib_config = {
    # === Settings for Environment ===
    "env": RllibDiscrete,
    "env_config": env_config,
    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    "num_workers": 2,
    # Number of environments to evaluate vector-wise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    "num_envs_per_worker": 8,
    # Divide episodes into fragments of this many steps each during rollouts.
    # Sample batches of this size are collected from rollout workers and
    # combined into a larger batch of `train_batch_size` for learning.
    #
    # For example, given rollout_fragment_length=100 and train_batch_size=1000:
    #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
    #   2. These fragments are concatenated and we perform an epoch of SGD.
    #
    # When using multiple envs per worker, the fragment size is multiplied by
    # `num_envs_per_worker`. This is since we are collecting steps from
    # multiple envs in parallel. For example, if num_envs_per_worker=5, then
    # rollout workers will return experiences in chunks of 5*100 = 500 steps.
    #
    # The dataflow here can vary per algorithm. For example, PPO further
    # divides the train batch into minibatches for multi-epoch SGD.
    "rollout_fragment_length": 200,
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_gpus_per_worker": 0 / 10,
    # === Settings for the Trainer process ===
    # Discount factor of the MDP.
    "gamma": 0.99,
    # The default learning rate of SGD.
    "lr": 0.00005,
    # Learning rate schedule.
    "lr_schedule": None,
    # Training batch size, if applicable. Should be >= rollout_fragment_length and
    # should be <= rollout_fragment_length * worker * envs/worker
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    "train_batch_size": 3200,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 512,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 10,
    # === PPO Specific ===
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    # should lay between 0.5 and 1.0
    "vf_loss_coeff": 0.9,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 1e-4,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.25,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # === Deep Learning Framework Settings ===
    # tf: TensorFlow (static-graph)
    # tf2: TensorFlow 2.x (eager)
    # tfe: TensorFlow eager
    # torch: PyTorch
    "framework": "tfe",
    # Enable tracing in eager mode. This greatly improves performance, but
    # makes it slightly harder to debug since Python code won't be evaluated
    # after the initial eager pass. Only possible if framework=tfe.
    "eager_tracing": True,
    # optimizer seems to be Adam by default at the moment
    # "optimizer": {},
    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    "explore": True,
    # Provide a dict specifying the Exploration object's config.
    "exploration_config": {
        # The Exploration class to use. In the simplest case, this is the name
        # (str) of any class present in the `rllib.utils.exploration` package.
        # You can also provide the python class directly or the full location
        # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        # EpsilonGreedy").
        "type": "StochasticSampling",
        # Add constructor kwargs here (if any).
    },
    # === Settings for Multi-Agent Environments ===
    "multiagent": {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        "policies": {
            "learned": (
                None,
                OBS_SPACE_AGENT,
                ACT_SPACE_AGENT,
                {
                    "model": {
                        # Share layers for value function. If you set this to True, it's
                        # important to tune vf_loss_coeff.
                        "vf_share_layers": True,
                        # === Built-in options ===
                        # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
                        # These are used if no custom model is specified and the input space is 1D.
                        # Number of hidden layers to be used.
                        "fcnet_hiddens": [256, 256],
                        # Activation function descriptor.
                        # Supported values are: "tanh", "relu", "swish" (or "silu"),
                        # "linear" (or None).
                        "fcnet_activation": "tanh",
                        # === Options for custom models ===
                        # Name of a custom model to use
                        "custom_model": "my_model",
                        # Extra options to pass to the custom classes. These will be available to
                        # the Model's constructor in the model_config field. Also, they will be
                        # attempted to be passed as **kwargs to ModelV2 models. For an example,
                        # see rllib/models/[tf|torch]/attention_net.py.
                        "custom_model_config": {"true_obs_shape": 7},
                        # Name of a custom action distribution to use.
                        "custom_action_dist": None,
                        # Custom preprocessors are deprecated. Please use a wrapper class around
                        # your environment instead to preprocess observations.
                        "custom_preprocessor": None,
                    },
                },
            ),
        },
        # Function mapping agent ids to policy ids.
        "policy_mapping_fn": lambda x: "learned",
    },
    "no_done_at_end": False,
    # "seed": tune.grid_search(seeds),
    "callbacks": MyCallbacks,
}
