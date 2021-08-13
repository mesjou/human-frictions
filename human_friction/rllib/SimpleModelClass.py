import ray
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from gym import spaces
from ray.rllib.utils.annotations import override
from typing import Dict
from ray.rllib.utils.typing import TensorType, List
import numpy as np



class SimpleModelMasked(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        true_obs_nbr = model_config["custom_model_config"]["true_obs_nbr"]
        self.action_embed_size = model_config["custom_model_config"]["nbr_choices"]

        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=(true_obs_nbr,)),
            action_space, self.action_embed_size,
            model_config, name + "_action_embedding"
        )
        self.register_variables(self.action_embed_model.variables())

    @override(TFModelV2)
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"].get("avail_actions",np.ones(self.action_embed_size))

        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
           "obs": input_dict["obs"]["true_obs"]})
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector,
            axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
