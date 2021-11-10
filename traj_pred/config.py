import keras as K 
import numpy as np 

class Config():
	def __init__(self):
		self.data_frequency=3
		self.predict_time=5
		# n_features = 

		self.guassian_size_x = 30
		self.guassian_size_y = 30

		self.input_dim=[self.guassian_size_x,self.guassian_size_y, 1] 
		self.input_num = self.data_frequency*self.predict_time # only consider 5 seconds, not include further history points

		self.output_dim= self.data_frequency*self.predict_time
		self.batch_size = 32