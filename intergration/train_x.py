import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from model import _build_model, _build_simple_model, _build_simple_model2
from model_reduced import _build_model, _build_simple_model, _build_simple_model2, _build_reduced_model, _build_reduced_model2
from corex import Graph

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"

gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus)

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.9  #0.99, Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.05  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
	epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10

# # Use the Baseline Atari environment because of Deepmind helper functions
# env = make_atari("BreakoutNoFrameskip-v4")
# # Warp the frames, grey scale, stake four frame and scale to smaller ratio
# env = wrap_deepmind(env, frame_stack=True, scale=True)
# env.seed(seed)
env=Graph()

num_actions = 35

# The first model makes the predictions for Q-values which are used to
# make a action.
model = _build_reduced_model2(num_actions)
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = _build_reduced_model2(num_actions)

# optimizer = keras.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)
optimizer = keras.optimizers.Adagrad(learning_rate=0.003)
# optimizer = keras.optimizers.Adadelta(learning_rate=0.003)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output, warm-up
# epsilon_random_frames = 50000
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 300000
# Train the model after a few actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

while episode_count<2000000:  # Run until solved
	# state = np.array(env.reset())
	state = env.reset()
	# print('==== episode====:', episode_count)
	episode_reward = 0

	for timestep in range(max_steps_per_episode):
		# env.render(); Adding this line would show the attempts
		# of the agent in a pop up window.
		frame_count += 1

		# Use epsilon-greedy for exploration
		if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
			# Take random action
			action = np.random.choice(num_actions)
			# print("action chosen", action)
		else:
			# Predict action Q-values
			# From environment state
			# subject, veh, ped, cyc, edge
			subject_tensor = tf.convert_to_tensor(state[0])
			subject_tensor = tf.expand_dims(subject_tensor, 0)

			veh_tensor = tf.convert_to_tensor(state[1])  
			veh_tensor = tf.expand_dims(veh_tensor, 0)

			ped_tensor = tf.convert_to_tensor(state[2])  
			ped_tensor = tf.expand_dims(ped_tensor, 0)

			cyc_tensor = tf.convert_to_tensor(state[3])  
			cyc_tensor = tf.expand_dims(cyc_tensor, 0)

			edge_tensor = tf.convert_to_tensor(state[4])   
			edge_tensor = tf.expand_dims(edge_tensor, 0) 

			# state_tensor = tf.expand_dims(state_tensor, 0)
			state_tensor = [subject_tensor, veh_tensor, ped_tensor, cyc_tensor, edge_tensor]
			action_probs = model(state_tensor, training=False)
			# Take best action
			action = tf.argmax(action_probs[0]).numpy()

		# Decay probability of taking random action
		epsilon -= epsilon_interval / epsilon_greedy_frames
		epsilon = max(epsilon, epsilon_min)

		# Apply the sampled action in our environment
		# print("action chosen", action)
		state_next, reward, done, _ = env.step(action)

		# print("reward at time", reward, timestep, done, _)
		# state_next = np.array(state_next)

		episode_reward += reward
		# print(episode_reward)

		# Save actions and states in replay buffer
		action_history.append(action)
		state_history.append(state)
		state_next_history.append(state_next)
		done_history.append(done)
		rewards_history.append(reward)
		state = state_next

		# Update every fourth frame and once batch size is over 32
		if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

			# Get indices of samples for replay buffers
			indices = np.random.choice(range(len(done_history)), size=batch_size)

			# Using list comprehension to sample from replay buffer
			subject_sample, veh_sample, ped_sample, cyc_sample, edge_sample =[],[],[],[],[]
			for i in indices:
				shi = state_history[i]
				subject_sample.append(shi[0])
				veh_sample.append(shi[1])
				ped_sample.append(shi[2])
				cyc_sample.append(shi[3])
				edge_sample.append(shi[4])
			subject_sample=np.array(subject_sample)
			veh_sample = np.array(veh_sample)
			ped_sample = np.array(ped_sample)
			cyc_sample = np.array(cyc_sample)
			edge_sample = np.array(edge_sample)
			state_sample = [subject_sample, veh_sample, ped_sample, cyc_sample, edge_sample] # state sample may be transformed to the mid-layer output

			subject_next_sample, veh_next_sample, ped_next_sample, cyc_next_sample, edge_next_sample =[],[],[],[],[]
			for i in indices:
				shi = state_next_history[i]
				subject_next_sample.append(shi[0])
				veh_next_sample.append(shi[1])
				ped_next_sample.append(shi[2])
				cyc_next_sample.append(shi[3])
				edge_next_sample.append(shi[4])
			subject_next_sample=np.array(subject_next_sample)
			veh_next_sample = np.array(veh_next_sample)
			ped_next_sample = np.array(ped_next_sample)
			cyc_next_sample = np.array(cyc_next_sample)
			edge_next_sample = np.array(edge_next_sample)
			state_next_sample = [subject_next_sample, veh_next_sample, ped_next_sample, cyc_next_sample, edge_next_sample]

			# for ele in state_sample:
			# 	print(ele.shape, 'ele')
			# for x in state_next_sample:
			# 	print(x.shape, 'x')
			# state_next_sample = [state_next_history[i] for i in indices]
			rewards_sample = [rewards_history[i] for i in indices]
			action_sample = [action_history[i] for i in indices]
			done_sample = tf.convert_to_tensor(
				[float(done_history[i]) for i in indices]
			)

			# Build the updated Q-values for the sampled future states
			# Use the target model for stability
			future_rewards = model_target.predict(state_next_sample) # get q value 
			# Q value = reward + discount factor * expected future reward
			updated_q_values = rewards_sample + gamma * tf.reduce_max(
				future_rewards, axis=1
			)

			# If final frame set the last value to -1
			# updated_q_values = updated_q_values * (1 - done_sample) - done_sample  # to be updated 

			# Create a mask so we only calculate loss on the updated Q-values
			masks = tf.one_hot(action_sample, num_actions)

			with tf.GradientTape() as tape:
				# Train the model on the states and updated Q-values
				q_values = model(state_sample)

				# Apply the masks to the Q-values to get the Q-value for action taken
				q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
				# Calculate loss between new Q-value and old Q-value
				loss = loss_function(updated_q_values, q_action)
				# print("shape, loss, updated_q_values, q_values, q_action", loss, updated_q_values.shape, q_values.shape, q_action.shape)
			# Backpropagation
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))

		if frame_count % update_target_network == 0:
			# update the the target network with new weights
			model_target.set_weights(model.get_weights())  
			# Log details
			template = "running reward: {:.2f} at episode {}, frame count {}"
			print(template.format(running_reward, episode_count, frame_count))

		# Limit the state and reward history
		if len(rewards_history) > max_memory_length:
			del rewards_history[:1]
			del state_history[:1]
			del state_next_history[:1]
			del action_history[:1]
			del done_history[:1]

		if done:
			break

	# Update running reward to check condition for solving
	episode_reward_history.append(episode_reward)
	if len(episode_reward_history) > 1000000:
		del episode_reward_history[:1]
	reward_len = len(episode_reward_history)
	if reward_len<=1000:
		running_reward = np.mean(episode_reward_history)
	else:
		running_reward = np.mean(episode_reward_history[reward_len-1000:])

	episode_count += 1
	if episode_count%100 ==0:
		print("modelxf_universal episode %d running reward %f" %(episode_count, running_reward))
	if episode_count%5000==0:
		np.save('modelxf_universal_episode_history', episode_reward_history)
		print("reward history saved")
		try:
			model.save('reduced_modelxf_universal') # only one task
			model_target.save('reduced_target_modelxf_universal')
		except Exception as e:
			print(e)

	# if running_reward > 200:  # Condition to consider the task solved
		# print("Solved at episode {}!".format(episode_count))
		# break
