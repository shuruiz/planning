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
from utils import Gibbs_sampling, get_smoothness, get_distance
from config import Config
from visualizer import plot_scene_on_grid


def compute_vweight(target, veh):
	"""
	computes the edge weights between target and veh
	"""
	# here only one pos, not all future positions
	try:
		d =get_distance(target.pos, veh.get_pos(target.t))
	except:
		print("raise expection compute_vweight")
		return math.exp(veh.risk)
	return math.exp(veh.risk)*d
	
def compute_pcweight(target, pc):
	"""
	computes the edge weights between target and ped/cyc
	"""
	try:
		d =get_distance(target.pos, veh.get_pos(target.t))
	except:
		print("raise expection compute_pcweight")
		return 1
	return d



class SceneGenerator():
	"""
	generate a scene
	"""
	def __init__(self):
		self.pxy= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pxy.pickle",'rb'))
		self.pxz= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pxz.pickle",'rb'))
		self.pyzx= pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_Pyzx.pickle",'rb'))

		self.veh_model = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_veh_model.pickle",'rb'))
		self.ped_model = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_ped_model.pickle",'rb'))
		self.cyc_model = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_cyc_model.pickle",'rb'))

		self.poolv = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_veh_pool.pickle",'rb'))
		self.poolp = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_ped_pool.pickle",'rb'))
		self.poolc = pickle.load(open("/home/lab1/repo/planning/saved_gibbs/8kfb_cyc_pool.pickle",'rb'))
	def generate(self):
		sampling_res = Gibbs_sampling(max_scene=1, Pxy=pxy, Pxz=pxz, Pyzx=pyzx, poolv=poolv, poolp=poolp, poolc=poolc,\
							 veh_model = veh_model, ped_model=ped_model, cyc_model=cyc_model,\
							 max_n_veh=10, max_n_ped=5,max_n_cyc=5)
		return sampling_res[0]


class TargetNode():
	"""
	build target node
	"""
	def __init__(self, traj):
		self.start=traj[10] # position at 5th second
		self.goal=traj[-1] # last point
		self.pos=self.start
		self.task=np.array(self.start,self.goal)
  
		self.a,self.v = self._get_av(traj[8], traj[9], traj[10])
		self.theta = self._get_theta(traj[9], traj[10])
		self.t=0
		
		self.history=[]
		self.history.append([self.t, self.pos, self.a, self.theta])
		
		
	def _get_av(self, prev2, prev, curr):
		v10 = np.linalg.norm(curr - prev)/0.5
		v9 = np.linalg.norm(prev - prev2)/0.5
		a = (v10-v9)/0.5
		return a, v10
	
	def _get_theta(self, prev,curr):
		try:
			tan = (curr[1]-prev[1])/(curr[0]-prev[0])
			return math.atan(tan)
		except:
			return 90

	def act(self, action_pair):
		_a, _theta = action_pair[0], action_pair[1] # can change the action to  [delta_a, delta_theta]
		ds = self.v * 0.5+ 0.125* _a 
		dx = ds*math.cos(_theta)
		dy = ds*math.sin(_theta)
		self.pos = [self.pos[0]+dx, self.pos[1]+dy]
		self.t+=1
		self.history.append([self.t, self.pos, _a, _theta])

		
class Node():
	"""
	build other nodes
	"""
	def __init__(self,traj, pred, agent='veh',risk=0.5):
		self.traj=traj
		self.pred=pred
		self.agent=agent
		self.t=0
		if agent=='veh':
			self.risk=risk
		else:
			# if agent==veh, there is risk, otherwise the risk is not meaningful
			self.risk=-1
	def get_pos(self,t):
		temp = np.vstack((self.traj, self.pred))
		return temp[t+10,:]


class Graph():
	"""
	construct a graph representing env
	a scene, prediction models, and lane_boundary are needed to initialize
	"""
	def __init__(self, models, lane_boundary):
		self.gen_scene=SceneGenerator()
		self.sample=self.gen_scene.generate()
		self.graph={}
		self.vmodel=models[0]
		self.pmodel=models[1]
		self.cmodel=models[2]
		self.riskmodel=models[3]
		self.lane_boundary=lane_boundary
		
		self.action_dict = list(itertools.product(range(-3,3,0.1), range(0,180,3)))
		
		self._build_graph_
		
	def _build_graph_(self):
		subject = self.sample['veh'][0]
		self.target= TargetNode(subject)
		self.env_veh=[]
		self.env_ped=[]
		self.env_cyc=[]
		self.edges=[]
		
		# build nodes/vertices
		for i in range(1, len(self.sample['veh'])):
			traj = self.sample['veh'][i]
			pred_traj = pred_veh(self.vmodel, traj)
			risk = pred_risk(self.riskmodel, traj)
			node = Node(traj,pred_traj,'veh',risk)
			self.env_veh.append(node)
			
		
		for traj in samples['cyc']:
			pred_traj = pred_cyc(self.cmodel,traj)
			node=Node(traj, pred_traj,'cyc')
			self.env_cyc.append(node)
			
	
		for traj in samples['ped']:
			pred_traj = pred_cyc(self.pmodel,traj)
			node=Node(traj, pred_traj,'ped')
			self.env_ped.append(node)
		
		self._sort_env()
		self._update_edges()
		# build edges
	def _sort_env(self):
		v_distance_key=[]
		for veh in self.env_veh:
			d = get_distance(self.target.pos, veh.get_pos(self.target.t))
			v_distance_key.append(d)
		self.env_veh=[x for _, x in sorted(zip(v_distance_key, self.env_veh))]

		p_distance_key=[]
		for ped in self.env_ped:
			d=get_distance(self.target.pos, ped.get_pos(self.target.t))
			p_distance_key.append(d)
		self.env_ped=[x for _, x in sorted(zip(p_distance_key, self.env_ped))]

		c_distance_key=[]
		for cyc in self.env_cyc:
			d=get_distance(self.target.pos, cyc.get_pos(self.target.t))
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
		subject =np.array([self.target.start, self.target.goal, self.target.pos[0],self.target.pos[1],self.target.a, self.target.theta, self.target.t])
		veh=[]
		v_count=0
		v_edge=[]
		for node in self.env_veh:
			veh.append(node.traj)
			v_edge.append(self.edges[count])
			count+=1
			if count>5:
				break
		if count<5:
			pad = np.zeros((5-count, 10, 2))
			veh = np.vstack((np.array(veh), pad))
			for _ in range(5-count):
				v_edge.append(0)

		p_count=0
		ped=[]
		p_edge=[]
		for node in self.env_ped:
			ped.append(node.traj)
			p_edge.append(self.edges[p_count+len(self.env_veh)])
			p_count+=1
			if p_count>3:
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
			if c_count>3:
				break
		if c_count<3:
			pad = np.zeros((3-c_count, 10, 2))
			cyc = np.vstack((np.array(cyc), pad))
			for _ in range(3-c_count):
				c_edge.append(0)

		edge = v_edge + p_edge + c_edge
		return [subject, veh, ped, cyc, np.array(edge)]


	
	def _take_action(self, policy):
		
		# action = self.take_action(policy)
		# return action
		self._sort_env()
		self._update_edges()
		
	
	def is_crash(self):
		"""
		check if crash happens when graph is updated, or hit lane_boundary inside intersection
		"""
		pass

	def step(self, action, policy):
		"""
		update graph, move to next state
		action space [-3, 3] * [0,180] => 60*180 action space  # a, theta are delta
		"""
		acc, theta = self.action_dict[action]
		self.target.a = self.target.a+acc
		self.target.theta=self.target.theta+theta

		distance = self.target.v*0.5+0.5*self.target.a*0.25
		radian = self.target.theta*math.pi/180
		dx, dy = distance*math.cos(radian), distance*math.sin(radian)
		
		#update state
		self.target.pos =[self.target.pos[0]+dx, self.target.pos[1]+dy]
		self.target.v = self.target.v+self.target.a*0.5
		self.target.t +=1 # plan step move forward

		self.target.history.append([self.target.t, self.target.pos, self.target.a, self.target.theta])
		# update the state
		self._sort_env()
		self._update_edges()  #maybe put to later
		distance_to_goal = np.linalg.norm(self.target.pos - self.target.goal, axis=1)
		
		# compute reward
		# stop crateria: #step longer than 5 seconds, reached goal, crash with other agents
		state_next = self.wrap_nn_input()
		reward, info = self._get_reward()

		if  info == 'reach_goal' or info =='time_out' or info== 'crash'
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
		pass

	def reset(self):
		self.sample=self.gen_scene.generate()
		self.graph={} 
		self._build_graph_
		return self.wrap_nn_input()


class Env():
	def __init__(self,n_actions=4320):
		self.n_actions= n_actions
		self.action_map={}

	def step(self, action):
		# return obs, reward, done, info
		pass

	def get_inputs():
		"""
		prepare the observation to model-ready inputs
		"""
		pass
	def reset():
		# build a new graph
		pass

