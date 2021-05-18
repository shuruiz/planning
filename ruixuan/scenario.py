import numpy as np
import os

from l5kit.geometry import *
from l5kit.dataset.select_agents import *
from l5kit.sampling import *
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from IPython.display import display, clear_output
from images2gif import writeGif
import PIL
import time
import math


def is_lane(elem, map_api):
    return elem.element.HasField("lane")


def get_lanes(map_api):
    return [elem for elem in map_api.elements if is_lane(elem, map_api)]


def lane_check(centroid: np.ndarray, lane_id: str, map_api: object) -> bool:
    point = Point(centroid[0],centroid[1])
    boundary = map_api.get_lane_coords(lane_id)
    left_boundary = boundary['xyz_left']
    right_boundary = boundary['xyz_right']
    left_boundary_point = [(line[0],line[1])for line in left_boundary]
    right_boundary_point = [(line[0],line[1])for line in reversed(right_boundary)]
    polygon = Polygon(left_boundary_point+right_boundary_point)
    
    return polygon.contains(point)


def current_lane(centroid: np.ndarray, Lane: dict, map_api: object) -> str:

    for key, value in Lane.items():
        if lane_check(centroid, key, map_api):
            return key


"""
For a given frame, check the current lane and find its left lane and right lane. For all valid agents in this frame, return the number of
agents in each lane. 

Mostly, if there exists at least one agent in the same lane of AV, check its time headway at this frame to determine if it is following AV

Having known the number of agents in lane at the current farme, check the chane of number at a fixed frequency. The most relevant case is that
the number in the same lane change with the same agents (smae track_id). The chamnge means that some agents change their lane or the agents in
other lanes change into this lane.

By using track_id, the agents can be further determined if they have changed the lane or just disappear from AV's observation window.

"""

# Before that, an additional step need to do: that is for a given lane, find all possible adjacent lanes

def all_adjacent_lanes(current_lane_id: str, Lane: dict, map_api: object) -> Tuple[list,list]:
    lanes_on_left = []
    lanes_on_right = []
    adjacent_left_lane = map_api.id_as_str(Lane[current_lane_id].adjacent_lane_change_left)
    adjacent_right_lane = map_api.id_as_str(Lane[current_lane_id].adjacent_lane_change_right)
    
    while adjacent_left_lane:
        lanes_on_left.append(adjacent_left_lane)
        next_left_lane =map_api.id_as_str(Lane[adjacent_left_lane].adjacent_lane_change_left)
        adjacent_left_lane = next_left_lane
    
    while adjacent_right_lane:
        lanes_on_right.append(adjacent_right_lane)
        next_right_lane =map_api.id_as_str(Lane[adjacent_right_lane].adjacent_lane_change_right)
        adjacent_right_lane = next_right_lane
        
    return lanes_on_left, lanes_on_right



"""
The first step is to filter the valid agents in a given frame within a distance and angular distance, also with non zero velocity

"""

# rewrite the in_av_distance function from l5kit.dataset.select_agents.py       
def in_AV_distance(av_translation: np.ndarray, agent_centroid: np.ndarray, th: float) -> bool:
    return np.linalg.norm(av_translation[:2] - agent_centroid, axis=1) < th


def valid_agents_mask(frame_interval: list, zarr_dataset: ChunkedDataset) -> Tuple[dict,slice]:
    
    future_step = 5
    distance_th = 60
    angular_th = 1
    future_num = int((frame_interval[1]-frame_interval[0])/future_step)
    interval_slice = get_future_slice(frame_interval[0], future_num, future_step)
    frames_agent_index_interval = zarr_dataset.frames[interval_slice]['agent_index_interval']
    frames_rotation = zarr_dataset.frames[interval_slice]['ego_rotation']
    frames_translation = zarr_dataset.frames[interval_slice]['ego_translation']
    
    
    # all needed information from valid agents in each checking frame and will be returned
    frames_agent_centroid = []
    frames_agent_velocity = []
    frames_agent_track_id = []
    frames_mask = []
    
    print('Finish the first part    ')
    for idx, agents_index_interval in enumerate(frames_agent_index_interval):
        
        agents_interval = [agents_index_interval[0],agents_index_interval[1]]
        agents_velocity = zarr_dataset.agents[agents_index_interval[0]:agents_index_interval[1]]['velocity']
        # velocity non-zero
        velocity_mask = (agents_velocity != np.array([0,0]))[:,0]   
        agent_mask = np.where(velocity_mask == True)[0] + agents_index_interval[0]
        
        agnets_yaw = zarr_dataset.agents[agent_mask]['yaw']
        # within angular distance
        yaw_mask = abs(agnets_yaw - rotation33_as_yaw(frames_rotation[idx])) < angular_th  
        agent_mask = agent_mask[np.where(yaw_mask == True)[0]]
            
        agents_centroid = zarr_dataset.agents[agent_mask]['centroid']
        distance_mask = in_AV_distance(frames_translation[idx],agents_centroid,distance_th)
        agent_mask = agent_mask[np.where(distance_mask == True)[0]]
         
        valid_agents_track_id = zarr_dataset.agents[agent_mask] ['track_id']
        valid_agents_centroid = zarr_dataset.agents[agent_mask]['centroid']
        valid_agents_velocity = zarr_dataset.agents[agent_mask]['velocity']
        
        
        frames_agent_centroid.append(valid_agents_centroid)
        frames_agent_velocity.append(valid_agents_velocity)
        frames_agent_track_id.append(valid_agents_track_id)
    
        # append the frame_mask
        frames_mask.append(agent_mask)
    
    
    agent_info = {'frames_mask':frames_mask, 'frames_agent_centroid':frames_agent_centroid, 'frames_agent_velocity':frames_agent_velocity, 'frames_agent_track_id':frames_agent_track_id}                      
    
    return agent_info, interval_slice



"""
At each frame, with valid agents, the next step is to check the number of agents on each lane, and their relative position to AV
The return result will be a dictionary

"""
def time_headway_check(agent_centroid: np.ndarray, agent_velocity: np.ndarray, centroid_AV: np.ndarray) -> bool:
    time_th = 4
    car_following = False
    
    for idx, centroid in enumerate(agent_centroid):
        
        relative_pos = angle_between_vectors(agent_velocity[idx], centroid_AV-centroid) < 1
        distance = np.linalg.norm(centroid_AV-centroid)
        velocity = np.linalg.norm(agent_velocity[idx])
        time = distance/velocity < time_th
        
        if (relative_pos and time) or (distance < 20):
            car_following = True
            return car_following
    return car_following



def agent_in_other_lane(lane_number_dict: dict, current_lane: str) -> list:
    other_lane = list(lane_number_dict.keys())
    other_lane.remove(current_lane)
    agent_in_other_lane = []
    for key in other_lane:  
        agent_in_other_lane += lane_number_dict[key]
    
    
    # a list that contains all agent track_id
    return agent_in_other_lane
    


# the process object is frame-wise
def agent_lane_info(agent_info: dict, his_frame_info: dict, Lane: dict, centroid_AV: np.array, map_api: object) -> Tuple[np.array, dict]:
    
    current_lane_id = current_lane(centroid_AV, Lane, map_api)
    lanes_on_left, lanes_on_right = all_adjacent_lanes(current_lane_id, Lane, map_api)
    
    # then to determine the number of agents on each lane
    # first is to check if there are other agents on the same lane as AV's
    
    #initial all lane with 0
    all_lanes = lanes_on_left+[current_lane_id]+lanes_on_right
    # initialize a dictionary whose key is id of each lane and the value is an empty list
    lane_number  = dict.fromkeys(all_lanes,[])    

    for idx, centroid in enumerate(agent_info['centroid']):
        for lane_id in all_lanes:
            if lane_check(centroid, lane_id, map_api):
                lane_number[lane_id].append(idx)
     
    # the track_id of agents in each lane
#     current_track_id = dict(zip(all_lanes,[agent_info['agent_track_id'][lane_number[lane]] for lane in all_lanes]))

    
    single_lane = len(lanes_on_left)+len(lanes_on_right) == 0
    
    car_following =  bool(lane_number[current_lane_id]) and time_headway_check(agent_info['centroid'][lane_number[current_lane_id]], agent_info['velocity'][lane_number[current_lane_id]], centroid_AV)
    
    agent_other_lane = agent_in_other_lane(lane_number,current_lane_id)
    

    # need to check if the current lane exists in the previous checking frame, and the numbers of agents in this lane are the same  
    if  current_lane_id in his_frame_info['lane_number'].keys() and lane_number[current_lane_id].sort() != his_frame_info['lane_number'][current_lane_id].sort():
        # it means the number of agents in AV's lane changed, then we need to check if the change is caused by lane_changing
        # By checking the track_id on each lane in current and history frame, compare them
        
        # the current track_id on AV's lane
        diff_agent = list(set(lane_number[current_lane_id])^set(his_frame_info['lane_number'][current_lane_id]))
        
        # check if the diff_agent in other lanes of current frame. In fact, two possible cases: 
        
        for agent_idx in diff_agent:
            if (agent_idx in agent_other_lane) or (agent_idx in his_frame_info['agent_other_lane']):
                lane_changing = True
                break
                
    else:
        lane_changing = False

    current_frame_info = {'lane_number':lane_number, 'agent_other_lane':agent_other_lane}
    
    return np.array([single_lane,car_following,lane_changing]), current_frame_info



def initial_frame_info(agent_info: dict, Lane: dict, centroid_AV: np.array, map_api: object) -> dict:
    current_lane_id = current_lane(centroid_AV, Lane, map_api)
    lanes_on_left, lanes_on_right = all_adjacent_lanes(current_lane_id, Lane, map_api)
    
    # then to determine the number of agents on each lane
    # first is to check if there are other agents on the same lane as AV's
    
    #initial all lane with 0
    all_lanes = lanes_on_left+[current_lane_id]+lanes_on_right
    # initialize a dictionary whose key is id of each lane and the value is an empty list
    lane_number  = dict.fromkeys(all_lanes,[])    

    for idx, centroid in enumerate(agent_info['centroid']):
        for lane_id in all_lanes:
            if lane_check(centroid, lane_id, map_api):
                lane_number[lane_id].append(idx)
                
    agent_other_lane = agent_in_other_lane(lane_number,current_lane_id)
    
    current_frame_info = {'lane_number':lane_number, 'agent_other_lane':agent_other_lane}
    
    return current_frame_info



"""
Now the procedure is frame_interval based. For each frame_interval, we can determine what kinds of scenarios happened during this
period. Then we can further use these data to train the model.
"""

def character_label(info: dict, interval_slice: slice, Lane: dict, zarr_dataset: ChunkedDataset, map_api: object) -> Tuple[np.array, np.array]:

    centroid_AV = zarr_dataset.frames['ego_translation'][interval_slice][:,:2]
    centroid_iter = iter(centroid_AV)
    next(centroid_iter)

    agent_info = {'centroid':info['frames_agent_centroid'][0],'agent_track_id':info['frames_agent_track_id'][0],'velocity':info['frames_agent_velocity'][0]}
    his_frame_info = initial_frame_info(agent_info, Lane, centroid_AV[0], map_api)
    label_count = np.zeros(3)
    number_instance = len(info['frames_mask'])
    for idx, centroid in enumerate(centroid_iter, start=1):
        
        agent_info = {'centroid':info['frames_agent_centroid'][idx],'agent_track_id':info['frames_agent_track_id'][idx],'velocity':info['frames_agent_velocity'][idx]}  
        
        try:    
            label, frame_info = agent_lane_info(agent_info, his_frame_info, Lane, centroid, map_api)
            his_frame_info = frame_info
            label_count += (label+0)
        except:
            pass
        
    
    label_prob = label_count/(number_instance-1)

    return label_prob, label_count



def label_interval(label_prob: np.array) -> np.array:

    # determine if it is single-lane scenario

    if label_prob[0] > 0.5:
        label_lane = 1
    else:
        label_lane = 0

    # determine if it is car-following scenario

    if label_prob[1] > 0.1:
        label_following = 1
    else:
        label_following = 0

    # determine if it is lane-changing scenario

    if label_prob[2] > 0:
        lane_changing = 1
    else:
        lane_changing = 0

    label = np.array([label_lane, label_following, lane_changing])

    return label



def store_data_label(data: list, indicator: bool, Lane: dict, zarr_dataset: ChunkedDataset, map_api: object, file_name: str):
    # if indicator is 1, then the data is corridor without crosswalk
    # otherwise is corridor with crosswalk

    if indicator:
        suffix = ''
        os.chdir("C:\\Users\\zheng\\Desktop\\UMich\\Independent Study\\Codes\\python codes\\data\\label\\corridor")
    else:
        suffix = '_crosswalk'
        os.chdir("C:\\Users\\zheng\\Desktop\\UMich\\Independent Study\\Codes\\python codes\\data\\label\\corridor_crosswalk")

    label_output = np.zeros((len(data),3))
    label_count_output = np.zeros(3)

    for idx, instance in tqdm(enumerate(data)):
        info, interval_slice = valid_agents_mask(instance, zarr_dataset)

        try:
            label_prob, label_count = character_label(info, interval_slice, Lane, zarr_dataset, map_api)
            label_count_output += label_count
            label = label_interval(label_prob)
            label_output[idx,:] = label
        except:
            print('The error idx is',idx)
            label_output[idx,:] = np.array([-1,-1,-1])
            pass

    output = {'label': label_output, 'label_count':label_count_output}
    np.save(file_name + '_label'+'.npy', output)
