import os

import ray
from human_friction.rllib.rllib_env import RllibEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# seeds = list(range(1))
env_config = {
    "episode_length": 800,
    "n_agents": 4,
}

rllib_config = {
    # === Settings for Environment ===
    "env": RllibEnv,
    "env_config": env_config,
    # Whether to clip rewards during Policy's postprocessing.
    # None (default): Clip for Atari only (r=sign(r)).
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    # False: Never clip.
    # [float value]: Clip at -value and + value.
    # Tuple[value1, value2]: Clip at value1 and value2.
    "clip_rewards": None,
    # If True, RLlib will learn entirely inside a normalized action space
    # (0.0 centered with small stddev; only affecting Box components) and
    # only unsquash actions (and clip just in case) to the bounds of
    # env's action space before sending actions back to the env.
    "normalize_actions": True,
    # Whether to use "rllib" or "deepmind" preprocessors by default
    "preprocessor_pref": "deepmind",
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
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
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
    "vf_loss_coeff": 1.0,
    "model": {
        # Share layers for value function. If you set this to True, it's
        # important to tune vf_loss_coeff.
        "vf_share_layers": False,
        # === Built-in options ===
        # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
        # These are used if no custom model is specified and the input space is 1D.
        # Number of hidden layers to be used.
        "fcnet_hiddens": [256, 256],
        # Activation function descriptor.
        # Supported values are: "tanh", "relu", "swish" (or "silu"),
        # "linear" (or None).
        "fcnet_activation": "tanh",
        # Some default models support a final FC stack of n Dense layers with given
        # activation:
        # - Complex observation spaces: Image components are fed through
        #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
        #   everything is concated and pushed through this final FC stack.
        # - VisionNets (CNNs), e.g. after the CNN stack, there may be
        #   additional Dense layers.
        # - FullyConnectedNetworks will have this additional FCStack as well
        # (that's why it's empty by default).
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        # For DiagGaussian action distributions, make the second half of the model
        # outputs floating bias variables instead of state-dependent. This only
        # has an effect is using the default fully connected net.
        "free_log_std": False,
        # Whether to skip the final linear layer used to resize the hidden layer
        # outputs to size `num_outputs`. If True, then the last hidden layer
        # should already match num_outputs.
        "no_final_linear": False,
        # == LSTM ==
        # Whether to wrap the model with an LSTM.
        "use_lstm": False,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 20,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        "_time_major": False,
        # == Attention Nets (experimental: torch-version is untested) ==
        # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
        # wrapper Model around the default Model.
        "use_attention": False,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 1,
        # The input and output size of each transformer unit.
        "attention_dim": 64,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 1,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 32,
        # The memory sizes for inference and training.
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 32,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 0,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 0,
        # === Options for custom models ===
        # Name of a custom model to use
        "custom_model": None,
        # Extra options to pass to the custom classes. These will be available to
        # the Model's constructor in the model_config field. Also, they will be
        # attempted to be passed as **kwargs to ModelV2 models. For an example,
        # see rllib/models/[tf|torch]/attention_net.py.
        "custom_model_config": {},
        # Name of a custom action distribution to use.
        "custom_action_dist": None,
        # Custom preprocessors are deprecated. Please use a wrapper class around
        # your environment instead to preprocess observations.
        "custom_preprocessor": None,
    },
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # === Deep Learning Framework Settings ===
    # tf: TensorFlow (static-graph)
    # tf2: TensorFlow 2.x (eager)
    # tfe: TensorFlow eager
    # torch: PyTorch
    "framework": "tf",
    # Enable tracing in eager mode. This greatly improves performance, but
    # makes it slightly harder to debug since Python code won't be evaluated
    # after the initial eager pass. Only possible if framework=tfe.
    "eager_tracing": False,
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
    # "multiagent": {
    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    # "policies": {
    # "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {"fcnet_hiddens": [256, 256]}),
    # },
    # Function mapping agent ids to policy ids.
    # "policy_mapping_fn": lambda x: 'learned',
    # },
    # "seed": tune.grid_search(seeds),
}


def run(debug=True, iteration=200):
    stop = {"training_iteration": 2 if debug else iteration}
    tune_analysis = tune.run(
        PPOTrainer, config=rllib_config, stop=stop, checkpoint_freq=0, checkpoint_at_end=True, name="PPO_New_Keynes"
    )
    return tune_analysis


if __name__ == "__main__":
    ray.init()
    run(debug=False)
    ray.shutdown()
