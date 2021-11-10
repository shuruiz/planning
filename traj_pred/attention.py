from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate,TimeDistributed
import config 
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow


# class attention(Layer):
# 	def __init__(self, causal=False, **kwargs):
# 		super(attention, self).__init__(**kwargs)
# 		self.causal = causal
# 		self.supports_masking = True

# 	def _apply_scores(self, scores, value, scores_mask=None):
# 	"""Applies attention scores to the given value tensor.
# 	To use this method in your attention layer, follow the steps:
# 	* Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
# 	  `[batch_size, Tv]` to calculate the attention `scores`.
# 	* Pass `scores` and `value` tensors to this method. The method applies
# 	  `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
# 	  returns `matmul(attention_distribution, value).
# 	* Apply `query_mask` and return the result.
# 	Args:
# 	  scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
# 	  value: Value tensor of shape `[batch_size, Tv, dim]`.
# 	  scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
# 		`[batch_size, Tq, Tv]`. If given, scores at positions where
# 		`scores_mask==False` do not contribute to the result. It must contain
# 		at least one `True` value in each line along the last dimension.
# 	Returns:
# 	  Tensor of shape `[batch_size, Tq, dim]`.
# 	"""
# 	if scores_mask is not None:
# 		padding_mask = math_ops.logical_not(scores_mask)
# 	  # Bias so padding positions do not contribute to attention distribution.
# 		scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
# 		attention_distribution = nn.softmax(scores)
# 		return math_ops.matmul(attention_distribution, value)

# 	def call(self, inputs, mask=None):
# 		self._validate_call_args(inputs=inputs, mask=mask)
# 		q = inputs[0]
# 		v = inputs[1]
# 		k = inputs[2] if len(inputs) > 2 else v
# 		q_mask = mask[0] if mask else None
# 		v_mask = mask[1] if mask else None
# 		scores = self._calculate_scores(query=q, key=k)
# 		if v_mask is not None:
# 		  # Mask of shape [batch_size, 1, Tv].
# 		  v_mask = array_ops.expand_dims(v_mask, axis=-2)
# 		if self.causal:
# 		  # Creates a lower triangular mask, so position i cannot attend to
# 		  # positions j>i. This prevents the flow of information from the future
# 		  # into the past.
# 		  scores_shape = array_ops.shape(scores)
# 		  # causal_mask_shape = [1, Tq, Tv].
# 		  causal_mask_shape = array_ops.concat(
# 			  [array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]],
# 			  axis=0)
# 		  causal_mask = _lower_triangular_mask(causal_mask_shape)
# 		else:
# 		  causal_mask = None
# 		scores_mask = _merge_masks(v_mask, causal_mask)
# 		result = self._apply_scores(scores=scores, value=v, scores_mask=scores_mask)
# 		if q_mask is not None:
# 		  # Mask of shape [batch_size, Tq, 1].
# 		  q_mask = array_ops.expand_dims(q_mask, axis=-1)
# 		  result *= math_ops.cast(q_mask, dtype=result.dtype)
# 		return result

# 	def compute_mask(self, inputs, mask=None):
# 		self._validate_call_args(inputs=inputs, mask=mask)
# 		if mask:
# 		  q_mask = mask[0]
# 		  if q_mask is None:
# 			return None
# 		  return ops.convert_to_tensor(q_mask)
# 		return None

# 	def _validate_call_args(self, inputs, mask):
# 		"""Validates arguments of the call method."""
# 		class_name = self.__class__.__name__
# 		if not isinstance(inputs, list):
# 		  raise ValueError(
# 			  '{} layer must be called on a list of inputs, namely [query, value] '
# 			  'or [query, value, key].'.format(class_name))
# 		if len(inputs) < 2 or len(inputs) > 3:
# 		  raise ValueError(
# 			  '{} layer accepts inputs list of length 2 or 3, '
# 			  'namely [query, value] or [query, value, key]. '
# 			  'Given length: {}'.format(class_name, len(inputs)))
# 		if mask:
# 		  if not isinstance(mask, list):
# 			raise ValueError(
# 				'{} layer mask must be a list, '
# 				'namely [query_mask, value_mask].'.format(class_name))
# 		  if len(mask) != 2:
# 			raise ValueError(
# 				'{} layer mask must be a list of length 2, namely [query_mask, '
# 				'value_mask]. Given length: {}'.format(class_name, len(mask)))

# 	def get_config(self):
# 		config = {'causal': self.causal}
# 		base_config = super(attention, self).get_config()
# 		return dict(list(base_config.items()) + list(config.items()))

class Attention_Block(Layer):
	def __init__(self, **kwargs):
		super(Attention_Block, self).__init__(**kwargs)

	def build(self):
		pass

	def call(self, inputs):
		hidden_states = inputs
		hidden_size = int(hidden_states.shape[2])
		score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
		h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
		# score = dot([score_first_part, h_t], [2, 1], name='attention_score_layer')
		score = dot([score_first_part, h_t], [2, 1], name='attention_score_layer')

		attention_weights = Activation('softmax', name='attention_weight_layer')(score)
		# (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
		context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector_layer')
		pre_activation = concatenate([context_vector, h_t], name='attention_output_layer')
		attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector_layer')(pre_activation)
		return attention_vector



# def attention_block(hidden_states):
# 	"""
# 	Many-to-one attention mechanism for Keras.
# 	@param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
# 	@return: 2D tensor with shape (batch_size, 128)
# 	"""
# 	hidden_size = int(hidden_states.shape[2])

# 	# Inside dense layer
# 	#              hidden_states            dot               W            =>           score_first_part
# 	# (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
# 	# W is the trainable weight matrix of attention Luong's multiplicative style score

# 	score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)

# 	# score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
# 	#            score_first_part           dot        last_hidden_state     => attention_weights
# 	# (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)

# 	h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
# 	# score = dot([score_first_part, h_t], [2, 1], name='attention_score_layer')
# 	score = dot([score_first_part, h_t], [2, 1], name='attention_score_layer')

# 	attention_weights = Activation('softmax', name='attention_weight_layer')(score)
# 	# (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
# 	context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector_layer')
# 	pre_activation = concatenate([context_vector, h_t], name='attention_output_layer')
# 	attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector_layer')(pre_activation)
# 	return attention_vector
