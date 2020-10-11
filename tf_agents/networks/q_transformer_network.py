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

"""Transformer network for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow.keras import layers

from tf_agents.networks import network
from tf_agents.networks import transformer_encoding_network
from tf_agents.networks import q_network

from tf_agents.utils import nest_utils


@gin.configurable
class QTransformerNetwork(network.Network):
	"""Transformer Q network."""

	def __init__(
			self,
			input_tensor_spec,
			action_spec,
			d_model=None,
			num_heads=None,
			dff=None,
			num_layers=None,
			maximum_position_encoding=1000,
			dropout_rate=0.1,
			output_last_state=False,
			dtype=tf.float32,
			name='QTransformerNetwork',
	):
		"""Creates an instance of `QTransformerNetwork`.

		Args:
			input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
				input observations.
			action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
				actions.
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

		Raises:
			ValueError: If `action_spec` contains more than one action.
		"""
		q_network.validate_specs(action_spec, input_tensor_spec)
		action_spec = tf.nest.flatten(action_spec)[0]
		num_actions = action_spec.maximum - action_spec.minimum + 1

		self._encoder = transformer_encoding_network.TransformerEncodingNetwork(
				input_tensor_spec,
				d_model=d_model,
				num_heads=num_heads,
				dff=dff,
				num_layers=num_layers,
				maximum_position_encoding=maximum_position_encoding,
				dropout_rate=dropout_rate,
				output_last_state=output_last_state,
				dtype=dtype)

		self._q_value_layer = layers.Dense(
				num_actions,
				activation=None,
				kernel_initializer=tf.compat.v1.initializers.random_uniform(
						minval=-0.001, maxval=0.001),
				bias_initializer=tf.compat.v1.initializers.random_uniform(
						minval=-0.0001, maxval=0.0001),
				dtype=dtype,
				name='q_value/dense')

		super(QTransformerNetwork, self).__init__(
				input_tensor_spec=input_tensor_spec,
				state_spec=(),
				name=name)

		self._output_last_state = output_last_state

	def _create_look_ahead_mask(self, size):
		mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

		return mask  # (seq_len, seq_len)

	def call(self, observation, step_type=None, network_state=(), training=False):
		"""Runs the given observation through the network.
		Args:
			observation: The observation to provide to the network.
			step_type: The step type for the given observation. See `StepType` in
				time_step.py.
			network_state: A state tuple to pass to the network, mainly used by RNNs.
			training: Whether the output is being used for training.
		Returns:
			A tuple `(logits, network_state)`.
		"""

		# observation shape = [batch_size, seq_len, ...] or [batch_size, ...]
		num_outer_dims = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
		if num_outer_dims == 2:
			seq_length = observation.shape[1]
		else:
			seq_length = 1

		look_ahead_mask = self._create_look_ahead_mask(seq_length)  # (seq_len, seq_len)

		output, network_state = self._encoder(
			observation,
			step_type,
			network_state=network_state,
			training=training,
			mask=look_ahead_mask)

		q_value = self._q_value_layer(output, training=training)

		if not training and self._output_last_state:
			# Remove time dimension during inference/evaluation
			# and only output last element of output sequence to
			# get action of dimension (batch_size, ) instead of (batch_size, 1, )
			if num_outer_dims == 2:
				q_value = tf.squeeze(q_value, axis=1)

		return q_value, network_state