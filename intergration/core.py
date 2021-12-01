import sys 
from ../ruixuan.turning_scene import *
from l5kit.rasterization.rasterizer_builder import _load_metadata
import time
import pickle
import math
from collections import deque
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from tabulate import tabulate
from utils import Gibbs_sampling
from config import Config
from visualizer import plot_scene_on_grid


def compute_vweight(target, veh):
    """
    computes the edge weights between target and veh
    """
    pos = target.pos
    pass
    
def compute_pcweight(target, pc):
    """
    computes the edge weights between target and ped/cyc
    """
    pos = target.pos
    pass


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
        if agent=='veh':
            self.risk=risk
        else:
            # if agent==veh, there is risk, otherwise the risk is not meaningful
            self.risk=-1


class Graph():
	"""
	construct a graph
	a scene, prediction models, and lane_boundary are needed to initialize
	"""
    def __init__(self, scene, models, lane_boundary):
        self.sample=scene
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
            
        self._update_edges()
        # build edges
     
        
    def _update_edges(self):
        
        for node in self.env_veh:
            weight = compute_vweight(self.target, node)
            self.edges.append(weight)
            
        for node in self.env_cyc:
            weight = compute_pcweight(self.target, node)
            self.edges.append(weight)
            
        for node in self.env_ped:
            weight = compute_pcweight(self.target, node)
            self.edges.append(weight)
    
    def _take_action(self, policy):
        
        # action = self.take_action(policy)
        # return action
        self._update_edges()
        pass
    
    def is_crash(self):
        """
        check if crash happens when graph is updated, or hit lane_boundary inside intersection
        """
        pass

    def step(self, action, policy):
        """
        update graph, move to next state
        action space [-3, 3] * [0,180] => 60*180 action space
        """
        acc, theta = self.action_dict[action]
        distance = self.target.v*0.5+0.5*acc*0.25
        radian = theta*math.pi/180
        dx, dy = distance*math.cos(radian), distance*math.sin(radian)
        
        #update state
        self.target.pos =[self.target.pos[0]+dx, self.target.pos[1]+dy]
        self.target.v = self.target.v+acc*0.5
        self.target.a = acc
        self.target.theta=theta
        self.target.t +=1

        self.target.history.append([self.target.t, self.target.pos, self.target.a, self.target.theta])
        # update the state
        self._update_edges()  #maybe put to later
        distance_to_goal = np.linalg.norm(self.target.pos - self.target.goal, axis=1)
        
        # compute reward
        # stop crateria: #step longer than 5 seconds, reached goal, crash with other agents
        if  distance_to_goal<1:
            # update reward
            pass
        
        elif self.t>10:
            # update reward
             #return 'done'
            pass
        elif is_crash():  
            # this can be moved to update edge part by giving a huge loss, is the loss pass the threshold then done
            # update reward
            pass
        else:
            # update reward
            pass
        
    
    def _get_reward(self):
        """
        compute cost of the current state, A*sum_t sum_edge c_pi +B*c_goal +C* c_smoothness
        """
        pass


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

