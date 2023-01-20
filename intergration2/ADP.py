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

class ADP:
	def __init__(self, kmeans, prob):
		self.kmeans=kmeans
		self.prob = prob

	def predict(self, traj):
		feature = extract_feature(traj)
		category = self.kmeans.predict(feature)
		return self.prob[category]