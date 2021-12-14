import sys 
from ruixuan.turning_scene import *
from l5kit.rasterization.rasterizer_builder import _load_metadata
import time
import pickle
import math
from collections import deque
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from tabulate import tabulate
from utils import Gibbs_sampling, get_smoothness, get_distance_pt
from config import Config
from visualizer import plot_scene_on_grid
import itertools


def compute_vweight(target, veh):
	"""
	computes the edge weights between target and veh
	"""
	# here only one pos, not all future positions
	try:
		d =get_distance_pt(target.pos, veh.get_pos(target.t))
	except:
		print("raise expection compute_vweight")
		return math.exp(veh.risk)
	return math.exp(veh.risk)*d
	
def compute_pcweight(target, pc):
	"""
	computes the edge weights between target and ped/cyc
	"""
	try:
		# print('pcw', target.pos)
		# print(pc.get_pos(target.t))
		d =get_distance_pt(target.pos, pc.get_pos(target.t))
	except:
		print("raise expection compute_pcweight")
		return 1
	return d

def compute_jerkness(target):
	"""
	compute the jerkness of the trajectory, starting from existing traj
	"""
	history_a = np.array(target.history)[:,3]
	jerk=0
	for i in range(1,len(history_a)):
		jerk+= abs((history_a[i]-history_a[i-1])/0.5)
	return jerk/(len(history_a)-1)


class SceneGenerator():
	"""
	generate a scene
	"""
	def __init__(self):
		self.pxy= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pxy.pickle",'rb'))
		self.pxz= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pxz.pickle",'rb'))
		self.pyzx= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pyzx.pickle",'rb'))

		self.veh_gmm = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_veh_model.pickle",'rb'))
		self.ped_gmm = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_ped_model.pickle",'rb'))
		self.cyc_gmm = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_cyc_model.pickle",'rb'))

		self.poolv = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_veh_pool.pickle",'rb'))
		self.poolp = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_ped_pool.pickle",'rb'))
		self.poolc = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_cyc_pool.pickle",'rb'))
	def generate(self):
		sampling_res = Gibbs_sampling(max_scene=1, Pxy=self.pxy, Pxz=self.pxz, Pyzx=self.pyzx, poolv=self.poolv, poolp=self.poolp, poolc=self.poolc,\
							 veh_model = self.veh_gmm, ped_model=self.ped_gmm, cyc_model=self.cyc_gmm,\
							 max_n_veh=10, max_n_ped=5,max_n_cyc=5)
		return sampling_res[0]


class TargetNode():
	"""
	build target node
	"""
	def __init__(self, traj):
		self.traj = traj
		self.start=traj[10] # position at 5th second
		self.goal=traj[-1] # last point
		self.pos=self.start
		self.task=np.array([self.start,self.goal])
  
		self.a,self.v = self._get_av(traj[8], traj[9], traj[10])
		self.theta = self._get_theta(traj[9], traj[10])
		self.t=0
		
		self.history=[]
		self.history.append([self.t, self.pos[0],self.pos[1], self.a, self.theta])
		
		
	def _get_av(self, prev2, prev, curr):
		v10 = np.linalg.norm(curr - prev)/0.5
		v9 = np.linalg.norm(prev - prev2)/0.5
		a = (v10-v9)/0.5
		return round(a, 1), round(v10, 3)
	
	def _get_theta(self, prev,curr):
		try:
			tan = (curr[1]-prev[1])/(curr[0]-prev[0])
			return round(math.atan(tan), 0)
		except:
			return 90

	# def act(self, action_pair):
	# 	_a, _theta = action_pair[0], action_pair[1] # can change the action to  [delta_a, delta_theta]
	# 	ds = self.v * 0.5+ 0.125* _a 
	# 	dx = ds*math.cos(_theta)
	# 	dy = ds*math.sin(_theta)
	# 	self.pos = [self.pos[0]+dx, self.pos[1]+dy]
	# 	self.t+=1
	# 	self.history.append([self.t, self.pos, _a, _theta])

		
class Node():
	"""
	build other nodes
	"""
	def __init__(self,traj, agent='veh',risk=0.5):
		self.traj=traj[:10]
		
		self.agent=agent
		# self.t=0

		self.pred=np.array([traj[-1]]*10).reshape((10,2))
		if agent=='veh':
			self.risk=risk
			# self.pred=pred_veh(self.traj)
		elif agent=='ped':
			# if agent==veh, there is risk, otherwise the risk is not meaningful
			# self.pred=pred_ped(self.traj)
			self.risk=-1
		else:
			# self.pred=pred_cyc(self.traj)
			self.risk=-1

		self.projected=np.vstack((self.traj, self.pred))
	def get_pos(self,t):
		# temp = np.vstack((self.traj, self.pred))
		return self.projected[t+10,:]


class Graph():
	"""
	construct a graph representing env
	a scene, prediction models, and lane_boundary are needed to initialize
	"""
	def __init__(self, lane_boundary=[]):
		self.gen_scene=SceneGenerator()
		self.sample=self.gen_scene.generate()
		self.graph={}
		# self.vmodel=models[0]
		# self.pmodel=models[1]
		# self.cmodel=models[2]
		# self.riskmodel=models[3]
		self.lane_boundary=lane_boundary
		
		self.action_dict = list(itertools.product(np.round(np.arange(-3,3,0.1), decimals=1), np.round(np.arange(-90,90,3), decimals=0))) # delta a, delta theta
		# print(len(self.action_dict))
		
		# self._build_graph_()
		
	def _build_graph_(self):
		subject = self.sample['veh'][0]
		self.target= TargetNode(subject)
		# print('build target node')
		print('task', self.target.task)
		self.env_veh=[]
		self.env_ped=[]
		self.env_cyc=[]
		# self.edges=[]
		
		# build nodes/vertices
		for i in range(1, len(self.sample['veh'])):
			traj = self.sample['veh'][i]
			# pred_traj = pred_veh(self.vmodel, traj)
			# risk = pred_risk(self.riskmodel, traj)
			node = Node(traj,'veh')
			# print('build veh node')
			self.env_veh.append(node)
			
		
		for traj in self.sample['cyc']:
			# pred_traj = pred_cyc(self.cmodel,traj)
			node=Node(traj, 'cyc')
			# print('build cyc node')
			self.env_cyc.append(node)
			
	
		for traj in self.sample['ped']:
			# pred_traj = pred_cyc(self.pmodel,traj)
			node=Node(traj,'ped')

			# print('build ped node')
			self.env_ped.append(node)
		
		# print("target pos",self.target.pos)
		# print("==============")
		self._sort_env()
		self._update_edges()
		# build edges
	def _sort_env(self):
		v_distance_key=[]
		for veh in self.env_veh:
			d = get_distance_pt(self.target.pos, veh.get_pos(self.target.t))
			v_distance_key.append(d)
		self.env_veh=[x for _, x in sorted(zip(v_distance_key, self.env_veh))]

		p_distance_key=[]
		for ped in self.env_ped:
			d=get_distance_pt(self.target.pos, ped.get_pos(self.target.t))
			p_distance_key.append(d)
		self.env_ped=[x for _, x in sorted(zip(p_distance_key, self.env_ped))]

		c_distance_key=[]
		for cyc in self.env_cyc:
			d=get_distance_pt(self.target.pos, cyc.get_pos(self.target.t))
			c_distance_key.append(d)
		self.env_cyc=[x for _, x in sorted(zip(c_distance_key, self.env_cyc))]


	def _update_edges(self):
		self.edges=[]
		for node in self.env_veh:
			weight = compute_vweight(self.target, node)
			self.edges.append(weight)

		for node in self.env_cyc:
			weight = compute_pcweight(self.target, node)
			self.edges.append(weight)
			
		for node in self.env_ped:
			weight = compute_pcweight(self.target, node)
			self.edges.append(weight)


	def wrap_nn_input(self):
		subject =np.array([self.target.start[0], self.target.start[1], self.target.goal[0], self.target.goal[1], self.target.pos[0],self.target.pos[1],self.target.a, self.target.theta, self.target.t])
		veh=[]
		v_count=0
		v_edge=[]
		for node in self.env_veh:
			veh.append(node.traj)
			v_edge.append(self.edges[v_count])
			v_count+=1
			if v_count>=5:
				break
		if v_count<5:
			pad = np.zeros((5-v_count, 10, 2))
			veh = np.vstack((np.array(veh), pad))
			for _ in range(5-v_count):
				v_edge.append(0)

		p_count=0
		ped=[]
		p_edge=[]
		for node in self.env_ped:
			ped.append(node.traj)
			p_edge.append(self.edges[p_count+len(self.env_veh)])
			p_count+=1
			if p_count>=3:
				break
		if p_count<3:
			pad = np.zeros((3-p_count, 10, 2))
			ped = np.vstack((np.array(ped), pad))
			for _ in range(3-p_count):
				p_edge.append(0)

		c_count=0
		cyc=[]
		c_edge=[]
		for node in self.env_cyc:
			cyc.append(node.traj)
			c_edge.append(self.edges[c_count+len(self.env_veh)+len(self.env_ped)])
			c_count+=1
			if c_count>=3:
				break
		if c_count<3:
			pad = np.zeros((3-c_count, 10, 2))
			cyc = np.vstack((np.array(cyc), pad))
			for _ in range(3-c_count):
				c_edge.append(0)


		self.nn_edge = np.array(v_edge + p_edge + c_edge)
		self.nn_edge= np.expand_dims(self.nn_edge, axis=0)

		subject =  np.expand_dims(subject, axis=0)
		veh = np.array(veh)
		ped = np.array(ped)
		cyc = np.array(cyc)
		print("history", self.target.history)
		return [subject, veh, ped, cyc, self.nn_edge]


	
	# def _take_action(self, policy):
		
	# 	# action = self.take_action(policy)
	# 	# return action
	# 	self._sort_env()
	# 	self._update_edges()
		
	
	def is_crash(self):
		"""
		check if crash happens when graph is updated, or hit lane_boundary inside intersection
		"""
		return False

	def step(self, action):
		"""
		update graph, move to next state
		action space [-3, 3] * [0,180] => 60*180 action space  # a, theta are delta
		"""
		acc, theta = self.action_dict[action]
		print("acc theta", acc, theta)
		self.target.a = self.target.a+acc
		self.target.theta=self.target.theta+theta

		distance = self.target.v*0.5+0.5*self.target.a*0.25
		radian = self.target.theta*math.pi/180
		dx, dy = distance*math.cos(radian), distance*math.sin(radian)
		
		#update state
		self.target.pos =[round(self.target.pos[0]+dx, 3), round(self.target.pos[1]+dy, 3)]
		self.target.v = self.target.v+self.target.a*0.5
		self.target.t +=1 # plan step move forward


		self.target.history.append([self.target.t, self.target.pos[0], self.target.pos[1], round(self.target.a,1), round(self.target.theta, 0)])

		# update the state
		self._sort_env()
		self._update_edges()  

		
		# compute reward
		# stop crateria: #step longer than 5 seconds, reached goal, crash with other agents
		state_next = self.wrap_nn_input()
		reward, info = self._get_reward()

		if  info == 'reach_goal' or info =='time_out' or info== 'crash':
			# update reward
			return state_next, reward, True, info
		else:
			# update reward
			return state_next, reward, False, info
		
	
	def _get_reward(self):
		"""
		compute cost of the current state, A*sum_t sum_edge c_pi +B*c_goal +C* c_smoothness
		"""
		# distance_to_goal<3, self.t>=10, is_crash, info 
		# print("get reward",self.target.start, self.target.goal, self.target.t, self.target.pos)
		distance_to_goal = get_distance_pt(self.target.pos,  self.target.goal)
		# edge weights
		c_e = 0.01*np.sum(self.nn_edge)
		c_d = 0.8*distance_to_goal
		c_j  = compute_jerkness(self.target)
		print(c_e, c_d, c_j)
		r = -(c_e+c_d+c_j)
		# t
		if self.is_crash():
			return -99999, 'crash' 
		elif distance_to_goal<=1:
			return r, 'reach_goal'
		elif self.target.t>=10:
			return r, 'time_out'
		else:
			return r, 'feasible'


	def reset(self):
		self.sample=self.gen_scene.generate()
		self._build_graph_()
		return self.wrap_nn_input()



if __name__ =='__main__':
	env = Graph()
	s = env.reset()
	print("======")
	s_, r, done, info = env.step(2)
	print(r, done, info)
	print("======")
	s_, r, done, info = env.step(20)
	print(r, done, info)
	print("======")
	s_, r, done, info = env.step(60)
	print(r, done, info)
