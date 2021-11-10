import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np 
from copy import deepcopy

def discounted_l1(y_true,y_pred,discounted_factor = 0.99):
	"""
	copmpute the sum of discounted l1 in the prediction sequence.
	return a loss keras tensor
	"""
	distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

	print("distance shape:", distance.shape)
	batch=distance.shape[0]
	print("batch shape", batch)
	
	factor =[]

	for i in range(distance.shape[1]):
		ele = discounted_factor**i
		factor.append(ele)

	# b_factor= np.array([factor] *batch)
	b_factor= np.array(factor)
	print("b_factor", b_factor)

	discounted = distance *b_factor
	print("discounted", discounted.shape)

	# summing both loss values along batch dimension 
	loss = K.sum(distance, axis=1)       # (batch_size,)
	print("loss shape:", loss.shape)
	print("loss",loss)
	
	return loss


def max_displacement_error(y_true,y_pred,discounted_factor = 0.99):
	"""
	compute the max displacement in prediction and ground truth
	return a loss keras tensor
	"""
	distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

	print("distance shape:", distance.shape)
	batch=distance.shape[0]
	print("batch shape", batch)
	
	factor =[]

	for i in range(distance.shape[1]):
		ele = discounted_factor**i
		factor.append(ele)

	# b_factor= np.array([factor] *batch)
	b_factor= np.array(factor)
	print("b_factor", b_factor)

	discounted = distance *b_factor
	print("discounted", discounted.shape)

	# summing both loss values along batch dimension 
	loss = K.max(distance, axis=1)       # (batch_size,)
	print("loss shape:", loss.shape)
	print("loss",loss)
	
	return loss

def v3_displacement_error(y_true,y_pred,discounted_factor = 0.99):
	"""
	compute the max displacement in prediction and ground truth
	return a loss keras tensor
	"""
	distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

	# print("distance shape:", distance.shape)
	# batch=distance.shape[0]
	# print("batch shape", batch)
	
	# factor =[]

	# for i in range(distance.shape[1]):
	# 	ele = discounted_factor**i
	# 	factor.append(ele)

	# # b_factor= np.array([factor] *batch)
	# b_factor= np.array(factor)
	# print("b_factor", b_factor)

	# discounted = distance *b_factor
	# print("discounted", discounted.shape)

	# # summing both loss values along batch dimension 
	# loss = K.max(distance, axis=1)       # (batch_size,)
	# print("loss shape:", loss.shape)
	# print("loss",loss)
	
	return distance