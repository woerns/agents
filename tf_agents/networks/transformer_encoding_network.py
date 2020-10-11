# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer Encoding Network.

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # input_conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM cell
  [optional]: Dense  # output_fc_layer_params
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

from tf_agents.keras_layers import transformer_encoder_layer
from tf_agents.networks import network
from tf_agents.utils import nest_utils


@gin.configurable
class TransformerEncodingNetwork(network.Network):
  """Transformer network."""

  def __init__(
      self,
      input_tensor_spec,
      d_model=None,
      num_heads=None,
      dff=None,
      num_layers=None,
      maximum_position_encoding=1000,
      dropout_rate=0.1,
      output_last_state=False,
      dtype=tf.float32,
      name='TransformerEncodingNetwork',
  ):
    """Creates an instance of `TransformerEncodingNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
      d_model: Size of encoding vectors.
      num_heads: Number of attention heads.
      dff: Size of fully-connected feed-forward layer.
      num_layers: Number of transformer encoder layers.
      maximum_position_encoding: Maximum number of positions to encode.
      dropout_rate: Dropout rate of encoder layer.
      output_last_state: If true, the network will only output the last element
        of the predicted output sequence. This is typically desired during inference.
      dtype: The dtype to use by the layers of the network.
      name: A string representing name of the network.

    """

    super(TransformerEncodingNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self.d_model = d_model
    self.num_layers = num_layers

    self._output_last_state = output_last_state

    #         self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self._dense = tf.keras.layers.Dense(d_model, dtype=dtype)
    self._pos_encoding = self._positional_encoding(maximum_position_encoding,
                                            self.d_model)
    self._enc_layers = [transformer_encoder_layer.TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate, dtype=dtype)
                       for _ in range(num_layers)]
    self._dropout = tf.keras.layers.Dropout(dropout_rate, dtype=dtype)

  def _positional_encoding(self, max_position, d_model):

    def get_angles(pos, i, d_model):
      angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

      return pos * angle_rates

    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

  def call(self,
           observation,
           step_type,
           network_state=(),
           mask=None,
           training=False):
    """Apply the network.

    Args:
      observation: A tuple of tensors matching `input_tensor_spec`.
      step_type: A tensor of `StepType.
      network_state: (optional.) The network state.
      training: Whether the output is being used for training.

    Returns:
      `(outputs, network_state)` - the network output and next network state.

    Raises:
      ValueError: If observation tensors lack outer `(batch,)` or
        `(batch, time)` axes.
    """

    ### NOTE: step_type is currently not used to mask out invalid sequences

    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               self.input_tensor_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          observation)
      step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                        step_type)

    seq_len = tf.shape(observation)[1]

    # adding embedding and position encoding.
    #         x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x = self._dense(observation)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self._pos_encoding[:, :seq_len, :]

    x = self._dropout(x, training=training)

    for i in range(self.num_layers):
      x = self._enc_layers[i](x, training, mask)

    output = x # (batch_size, input_seq_len, d_model)

    if not training and self._output_last_state:
      # Only return last element of output sequence. Useful during inference.
      output = output[:, -1:, :]
      # # Remove time dimension from the output.
      # output = tf.squeeze(output, axis=1)  # (batch_size, d_model)

    if not has_time_dim:
      # Remove time dimension from the output.
      output = tf.squeeze(output, [1])

    return output, network_state
