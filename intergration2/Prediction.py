import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import glob
import os
from tslearn.clustering import TimeSeriesKMeans
import copy as cp
import random
import pickle
import multiprocessing
import time
from collections import Counter
import math

from utils import extract_feature
from Prediction_utils.pmp import PMP_model


class StepAttention():
	def __init__(self,veh_model_path, ped_model_path, cyc_model_path):
		self.veh_model = PMP_model(veh_model_path)
		self.ped_model = PMP_model(ped_model_path)
		self.cyc_model = PMP_model(cyc_model_path)

	def predict(self,obs, agent_type):
		"""
		predict future 5s trajectories
		"""
		if agent_type =='veh':
			return self.veh_model.predict(obs)
		elif agent_type=='ped':
			return self.ped_model.predict(obs)
		else:
			return self.cyc_model.predict(obs)

class ADP():
	def __init__(self, kmeans_path, prob_path):
		self.kmeans=pickle.load(open(kmeans_path, 'rb'))
		self.prob = pickle.load(open(prob_path, 'rb'))

	def predict(self, traj):
		"""
		giving a traj, return risk score
		"""
		feature = extract_feature(traj)
		category = self.kmeans.predict(feature)
		return self.prob[category]