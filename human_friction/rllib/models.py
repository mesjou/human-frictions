from typing import Dict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.typing import List, ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()


class FCNet(TFModelV2):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super(FCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        hiddens = model_config.get("fcnet_hiddens", [])
        activation = model_config.get("fcnet_activation")
        activation = get_activation_fn(activation)

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)),), name="observations")

        # Create layers 0 to second-last.
        # TODO akirosa added layer normalization as essential
        last_layer = inputs
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size, name="fc_{}".format(i), activation=activation, kernel_initializer=normc_initializer(1.0)
            )(last_layer)
            i += 1

        logits_out = tf.keras.layers.Dense(
            num_outputs, name="fc_out", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_layer)

        value_out = tf.keras.layers.Dense(
            1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01)
        )(last_layer)

        self.base_model = tf.keras.Model(inputs, [logits_out, value_out],)

        self._value_out = None

    @override(ModelV2)
    def forward(
        self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        output, self._value_out = self.base_model([input_dict["obs_flat"]])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        output = output + inf_mask

        return output, state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
