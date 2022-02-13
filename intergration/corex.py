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
from utils import Gibbs_sampling, get_smoothness, get_distance_pt, percent_left
from config import Config
from visualizer import plot_scene_on_grid
import itertools
from Trajectory import check_collision

collision = check_collision()

def compute_vweight(target, veh):
	"""
	computes the edge weights between target and veh
	"""
	# here only one pos, not all future positions
	# the smaller the distance, the larger the weights
	try:
		d =get_distance_pt(target.pos, veh.get_pos(target.t))
		
		w = math.exp(-0.01* d) # the further, the lower the stress

	except:
		print("raise expection compute_vweight")
		return math.exp(veh.risk)
	return math.exp(veh.risk)*w
	
def compute_pcweight(target, pc):
	"""
	computes the edge weights between target and ped/cyc
	"""
	try:
		# print('pcw', target.pos)
		# print(pc.get_pos(target.t))
		d =get_distance_pt(target.pos, pc.get_pos(target.t))
		w = math.exp(-0.01* d)

	except:
		print("raise expection compute_pcweight")
		return 1
	return w

def compute_jerkness(target):
	"""
	compute the jerkness of the trajectory, starting from existing traj
	"""
	history_a = np.array(target.history)[:,3]
	jerk=0
	for i in range(1,len(history_a)):
		jerk+= abs((history_a[i]-history_a[i-1])/0.5)
	return jerk/(len(history_a)-1)

def node_sort(s, nodes):
	sort_index = np.argsort(s)
	results=[]
	for ele in sort_index:
		results.append(nodes[ele])
	return results

# def check_lane(node, collision):
# 	return collision.check(node.pos, [node.history[-2][1], node.history[-2][2]])

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

		# self.pxy= pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_Pxy_partial.pickle",'rb'))
		# self.pxz= pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_Pxz_partial.pickle",'rb'))
		# self.pyzx= pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_Pyzx_partial.pickle",'rb'))

		# self.veh_gmm = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_veh_model_partial.pickle",'rb'))
		# self.ped_gmm = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_ped_model_partial.pickle",'rb'))
		# self.cyc_gmm = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_cyc_model_partial.pickle",'rb'))

		# self.poolv = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_veh_pool_partial.pickle",'rb'))
		# self.poolp = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_ped_pool_partial.pickle",'rb'))
		# self.poolc = pickle.load(open("/home/lab1/repo/planning/8kfb_gibbs/8kfb_cyc_pool_partial.pickle",'rb'))


		# self.tasks = pickle.load(open("/home/lab1/repo/planning/tasks/task.pickle",'rb'))
		self.tasks = None
	def generate(self):
		sampling_res = Gibbs_sampling(max_scene=1, Pxy=self.pxy, Pxz=self.pxz, Pyzx=self.pyzx, poolv=self.poolv, poolp=self.poolp, poolc=self.poolc,\
							 veh_model = self.veh_gmm, ped_model=self.ped_gmm, cyc_model=self.cyc_gmm,\
							 max_n_veh=10, max_n_ped=5,max_n_cyc=5, tasks=self.tasks)
		return sampling_res[0]


class TargetNode():
	"""
	build target node
	"""
	def __init__(self, traj):
		self.traj = traj
		self.start=traj[9] # position at 5th second
		self.goal=traj[-1] # last point
		self.pos=self.start
		self.task=np.array([self.start,self.goal])
		# print("task",self.task)
  
		self.a,self.v = self._get_av(traj[8], traj[9], traj[10])
		self.theta, _ = self._get_theta(traj[9], traj[10])
		self.guide = self._get_guide(self.traj)
		self.t=0
		
		self.history=[]
		self.history.append([self.t, self.pos[0],self.pos[1], self.a, self.theta])
		# print("reset target node", self.history)
		
		
	def _get_av(self, prev2, prev, curr):
		v10 = np.linalg.norm(curr - prev)/0.5
		v9 = np.linalg.norm(prev - prev2)/0.5
		a = (v10-v9)/0.5
		return a, (v10+v9)/2

	def _get_theta(self, prev,curr):
		diff_x = curr[0]-prev[0] 
		diff_y = curr[1]-prev[1]
		if diff_x >0 and diff_y >0:
			indicator=1
		elif diff_x<=0 and diff_y>=0:
			indicator=2
		elif diff_x<=0 and diff_y<=0:
			indicator=3
		else:
			indicator=4
		try:
			if diff_x !=0:
				tan = diff_y/diff_x
				# print("tan", tan, math.atan(tan)*180/math.pi)
				return math.atan(tan)*180/math.pi, indicator
			else:
				return 90, indicator
		except:
			return 90, indicator

	def _get_guide(self, traj):
		guide=[]
		for i in range(9,19):
			a,_ = self._get_av(traj[i-1], traj[i], traj[i+1])
			theta, indicator = self._get_theta(traj[i], traj[i+1])
			guide.append([a, theta, indicator])
		return guide

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

		# self.pred=np.array([traj[-1]]*10).reshape((10,2))
		self.pred = traj[10:]
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
		return self.projected[t+9,:]

	def get_traj(self, t):
		return self.projected[t:t+10]

class Graph():
	"""
	construct a graph representing env
	a scene, prediction models, and lane_boundary are needed to initialize
	"""
	def __init__(self):
		self.gen_scene=SceneGenerator()
		self.sample=self.gen_scene.generate()
		self.graph={}
		# self.vmodel=models[0]
		# self.pmodel=models[1]
		# self.cmodel=models[2]
		# self.riskmodel=models[3]
		# self.lane_boundary=lane_boundary
		self.target_dist_to_others=[]
		
		
		self.action_dict = list(itertools.product(np.round(np.arange(-0.1,0.15,0.05), decimals=2), np.round(np.arange(-3,4,1), decimals=0))) # delta a, delta theta
		# print(len(self.action_dict))
		

		# self._build_graph_()

		
	def _build_graph_(self):
		subject = self.sample['veh'][0]
		self.target= TargetNode(subject)
		# print('build target node')
		# print('task', self.target.task)
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
		# self.env_veh=[x for _, x in sorted(zip(v_distance_key, self.env_veh))]
		self.env_veh=node_sort(v_distance_key, self.env_veh)

		p_distance_key=[]
		for ped in self.env_ped:
			d=get_distance_pt(self.target.pos, ped.get_pos(self.target.t))
			p_distance_key.append(d)
		# self.env_ped=[x for _, x in sorted(zip(p_distance_key, self.env_ped))]
		self.env_ped = node_sort(p_distance_key, self.env_ped)

		c_distance_key=[]
		for cyc in self.env_cyc:
			d=get_distance_pt(self.target.pos, cyc.get_pos(self.target.t))
			c_distance_key.append(d)
		# self.env_cyc=[x for _, x in sorted(zip(c_distance_key, self.env_cyc))]
		self.env_cyc = node_sort(c_distance_key, self.env_cyc)

		self.target_dist_to_others = v_distance_key + p_distance_key + c_distance_key
		# print('dist to others', self.target_dist_to_others)

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
		# subject =np.array([self.target.start[0], self.target.start[1], self.target.goal[0], self.target.goal[1], self.target.pos[0],self.target.pos[1],self.target.a, self.target.theta, self.target.t])
		subject =np.array([self.target.goal[0], self.target.goal[1], self.target.pos[0],self.target.pos[1],self.target.a, self.target.theta, self.target.t])
		veh=[]
		v_count=0
		v_edge=[]
		st = self.target.t 
		for node in self.env_veh:
			veh.append(node.get_traj(st))
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
			ped.append(node.get_traj(st))
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
			cyc.append(node.get_traj(st))
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
		# print("history", self.target.history)
		return [subject, veh, ped, cyc, self.nn_edge]


	
	# def _take_action(self, policy):
		
	# 	# action = self.take_action(policy)
	# 	# return action
	# 	self._sort_env()
	# 	self._update_edges()


	
	# def is_crash(self):
	# 	"""
	# 	check if crash happens when graph is updated, or hit lane_boundary inside intersection
	# 	"""
	# 	return False

	def step(self, action):
		"""
		update graph, move to next state
		action space [-3, 3] * [0,180] => 60*180 action space  # a, theta are delta
		"""
		acc, theta = self.action_dict[action]
		
		guide_a, guide_theta, guide_indicator  = self.target.guide[self.target.t]
		# print("acc theta", acc, theta, guide_a, guide_theta)
		self.target.a = guide_a+acc
		self.target.theta=guide_theta+theta


		distance = self.target.v*0.5+0.5*self.target.a*0.25
		radian = self.target.theta*math.pi/180
		dx, dy = abs(distance*math.cos(radian)), abs(distance*math.sin(radian))
		#update state
		if guide_indicator==1: # zone 1
			self.target.pos =[self.target.pos[0]+dx, self.target.pos[1]+dy]
		elif guide_indicator==2:
			self.target.pos =[self.target.pos[0]-dx, self.target.pos[1]+dy]
		elif guide_indicator==3:
			self.target.pos =[self.target.pos[0]-dx, self.target.pos[1]-dy]
		else:
			self.target.pos =[self.target.pos[0]+dx, self.target.pos[1]-dy]

		
		self.target.v = self.target.v+self.target.a*0.5
		self.target.t +=1 # plan step move forward


		self.target.history.append([self.target.t, self.target.pos[0], self.target.pos[1], self.target.a, self.target.theta])

		# update the state
		self._sort_env()
		self._update_edges()  

		
		# compute reward
		# stop crateria: #step longer than 5 seconds, reached goal, crash with other agents
		state_next = self.wrap_nn_input()
		reward, info = self._get_reward()

		if  info == 'reach_goal' or info== 'crash' or info=='time_out':
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
		# distance_to_goal = percent_left(self.target)
		# edge weights
		c_e = np.max(self.nn_edge)
		c_d = distance_to_goal+0.01 # avoid divided by 0
		c_j  = compute_jerkness(self.target)
		
		# r = -c_e-c_d-c_j
		
		# r = -c_e/(0.1*c_d+1) - c_j/(0.1*c_d+1) + (1000/c_d)**2
		
		# r = -c_e - c_d # try only distance to goal and stress
		r = math.exp(-0.2*c_e) + math.exp(-1*c_j) # xd, xe, xf, xI
		# r = -0.2*c_e - c_j # xg

		# r = math.exp(-0.02*c_e) * math.exp(-0.1*c_j) * (1000/c_d)**2  # math.exp(-0.1*self.target.t) # may be discounted by time, stressor and jerkiness, boosted by distance to goal
		# print("cost c",c_e, c_d, c_j, r, self.target.a, self.target.theta, self.target.pos, self.target.goal, self.target.start, self.target.t)
		# print("and", -0.2*c_e, -0.1*c_j, self.target.v , math.exp(-0.2*c_e) , math.exp(-1*c_j) )
		if min(self.target_dist_to_others)<0.5:
			return 0, 'crash' 
		# elif collision.check([self.target.history[-2][1], self.target.history[-2][2]],self.target.pos):
		# 	return -9999999, 'crash' 
		# elif distance_to_goal<=5:
		elif distance_to_goal<=1:
			return r, 'reach_goal'
			# return 9999999, 'reach_goal'
		elif self.target.t>=10:
			return r, 'time_out'
		else:
			return r, 'feasible'


	def reset(self):
		# print("resetting.................")
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
