import matplotlib.pyplot as plt
import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.geometry.transform import *

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points, rotation33_as_yaw
from tqdm import tqdm
from collections import Counter
from l5kit.rasterization.rasterizer_builder import _load_metadata
from prettytable import PrettyTable

import os

# from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene
# from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from IPython.display import display, clear_output
import PIL


# import matlab
# import matlab.engine


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Scene(object):
    def __init__(self, dataset, Map_Api):
        self.dataset = dataset
        self.frames = self.dataset.frames
        self.scene_num = len(self.dataset.scenes)
        self.yaw_th = 1.25
        self.distance_th = 10
        # self.eng = matlab.engine.start_matlab()
        # self.eng.cd(r'D:\\GitHub\\Clone\\planning\\ruixuan\\utils') 
        self.turning_scenes = []
        self.turning_yaw_diff = {}
        self.tunring_frames = []
        self.map_api = Map_Api
        self.all_junctions = None
        self.all_lanes = None
        self.Junction_Lane = {}
        self.Lane = None
        self.Junctions = None
        self.junction_scene = {}
        self.junction_turning_scene = {}
        self.error_filtered_scene = []
        
        
    
    def is_element(self, elem, element_name):
        return elem.element.HasField(element_name)

    def get_elements(self, element_name):
        return [elem for elem in self.map_api.elements if self.is_element(elem, element_name)]

    
    def lane_check(self, centroid, lane_id):
        point = Point(centroid[0],centroid[1])
        boundary = self.map_api.get_lane_coords(lane_id)
        left_boundary = boundary['xyz_left']
        right_boundary = boundary['xyz_right']
        left_boundary_point = [(line[0],line[1])for line in left_boundary]
        right_boundary_point = [(line[0],line[1])for line in reversed(right_boundary)]
        polygon = Polygon(left_boundary_point+right_boundary_point)

        return polygon.contains(point)


    def current_lane(self, centroid):

        for key, value in self.Lane.items():
            if self.lane_check(centroid, key):
                return key
    

    def generate_info_from_MAP(self):
        self.all_junctions = self.get_elements("junction")
        self.all_lanes = self.get_elements("lane")
        self.Lane = {self.map_api.id_as_str(lane.id):lane.element.lane for lane in self.all_lanes}
        self.Junction = {self.map_api.id_as_str(junction.id):junction for junction in self.all_junctions}
        
        for junction in self.all_junctions:
            self.Junction_Lane[self.map_api.id_as_str(junction.id)] = []

            for lane in junction.element.junction.lanes:
                self.Junction_Lane[self.map_api.id_as_str(junction.id)].append(self.map_api.id_as_str(lane))

            self.Junction_Lane[self.map_api.id_as_str(junction.id)] = set(self.Junction_Lane[self.map_api.id_as_str(junction.id)])
        
        self.junction_scene = dict.fromkeys(list(self.Junction.keys()), [])
        self.junction_turning_scene = dict.fromkeys(list(self.Junction.keys()), {})
        for key in self.Junction.keys():
            self.junction_turning_scene[key] = {'Turning Left': [], 'Turning Right': []}
        
        
    def trajectory_visualize(self):
        plt.figure(figsize=(18,18))
        plt.scatter(self.frames["ego_translation"][:,0], self.frames["ego_translation"][:,1], marker='.')
        plt.scatter(self.frames["ego_translation"][self.tunring_frames,0], self.frames["ego_translation"][self.tunring_frames,1], marker='.', color='r')
        plt.axis("equal")
        plt.grid(which='both')
        axes = plt.gca()
        
    
    def scene_visualize(self, scene_idx, ego_dataset, cfg):
        
        indexes = ego_dataset.get_scene_indices(scene_idx)
        images = []

        for idx in indexes:

            data = ego_dataset[idx]
            im = data["image"].transpose(1, 2, 0)
            im = ego_dataset.rasterizer.to_rgb(im)
            target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
            center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        #     draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
            clear_output(wait=True)
            display(PIL.Image.fromarray(im))


    def scene_trajectory_visualize(self, scene_list):

        plt.figure(figsize=(18,18))
        plt.scatter(self.frames["ego_translation"][:,0], self.frames["ego_translation"][:,1], marker='.')

        for scene  in scene_list:
            frame_interval = self.dataset.scenes[scene]['frame_index_interval']
            plt.scatter(self.frames[frame_interval[0]:frame_interval[1]]["ego_translation"][:,0], self.frames[frame_interval[0]:frame_interval[1]]["ego_translation"][:,1], marker='.', color='r')


        plt.axis("equal")
        plt.grid(which='both')
        axes = plt.gca()





            
    def junction_visualize(self, junction_id):
        
        plt.figure(figsize=(18,18))
        plt.scatter(self.frames["ego_translation"][:,0], self.frames["ego_translation"][:,1], marker='.')
        
        lane_list = self.Junction_Lane[junction_id]
        
        for lane in lane_list:
            plt.scatter(self.map_api.get_lane_coords(lane)['xyz_left'][:,0], self.map_api.get_lane_coords(lane)['xyz_left'][:,1], marker='.', color='k')
            plt.scatter(self.map_api.get_lane_coords(lane)['xyz_right'][:,0], self.map_api.get_lane_coords(lane)['xyz_right'][:,1], marker='.', color='k')

        plt.axis("equal")
        plt.grid(which='both')
        axes = plt.gca()
    
        
    def turning_scenes_finding(self):
        
        bar = tqdm(range(self.scene_num))
        bar.set_description('Filtering the turning scenes: ')
        for scene in bar:

            start_frame = self.frames[self.dataset.scenes[scene]["frame_index_interval"][0]]
            finish_frame = self.frames[self.dataset.scenes[scene]["frame_index_interval"][1]-1]

            start_yaw = rotation33_as_yaw(start_frame["ego_rotation"])
            finish_yaw = rotation33_as_yaw(finish_frame["ego_rotation"])


            if abs(start_yaw-finish_yaw) > self.yaw_th:
                self.turning_scenes.append(scene)
                self.tunring_frames += list(range(self.dataset.scenes[scene]["frame_index_interval"][0],self.dataset.scenes[scene]["frame_index_interval"][1]))
                
    
   
    def chgpt_index_find(self, chgpt_loc, chgpt_dist, chgpt_num):
        rel_d = np.diff(chgpt_loc)
        
        while True in (rel_d < self.distance_th):
            for idx, d in enumerate(rel_d):
                if d < self.distance_th:
                    index = chgpt_loc[idx] if chgpt_dist[chgpt_loc[idx]]< chgpt_dist[chgpt_loc[idx+1]] else chgpt_loc[idx+1]
                    chgpt_dist[index] = 0
            chgpt_loc = np.argpartition(chgpt_dist, -chgpt_num)[-chgpt_num:]
            chgpt_loc.sort()
            # find the adjacent distance
            rel_d = np.diff(chgpt_loc)
            
            
        return chgpt_loc

    
            
    def scene_turning_frames(self, scene_idx):
        
        frames = list(range(self.dataset.scenes[scene_idx]["frame_index_interval"][0],self.dataset.scenes[scene_idx]["frame_index_interval"][1]))
        yaw = [rotation33_as_yaw(rotation) for rotation in self.dataset.frames['ego_rotation'][frames]]
        
        # extract key info from MATLAB results
        output_yaw = self.eng.mat2py(matlab.double(yaw), nargout=2)

        chgpt_dist_yaw = np.array(output_yaw[0])[0]
        chgpt_num_dist_yaw = np.array(output_yaw[1]).flatten()

        chgpt_num = np.argmax(chgpt_num_dist_yaw)
        
        
        if chgpt_num != 2:
            self.error_filtered_scene.append(scene_idx)
            
            return []
            
            
        else:
            chgpt_index = np.argpartition(chgpt_dist_yaw, -chgpt_num)[-chgpt_num:]
            chgpt_index.sort()

            chgpt_loc = self.chgpt_index_find(chgpt_index, chgpt_dist_yaw, chgpt_num)

            self.turning_yaw_diff[scene_idx] = yaw[chgpt_loc[0]] - yaw[chgpt_loc[1]]


            return list(range(chgpt_loc[0]+frames[0],chgpt_loc[1]+frames[0],3))
    
    
    
    
    def lanes_at_turning(self, scene_turning_frames):
        
        lanes = []

        for frame in scene_turning_frames:
            lanes.append(self.current_lane(self.frames[frame]['ego_translation'][:2]))

        lanes = set(lanes)

        return lanes
    
    
    
    
    def junction_id_find(self, lanes):
        junction_id = []

        
        for key in self.Junction_Lane.keys():

            if len(self.Junction_Lane[key]&lanes) > 0:
                junction_id.append(key)
                return junction_id
                
                
        return junction_id
    
    
    
    
    def junction_scene_find(self, scene_idx, junction_id):
    
        self.junction_scene[junction_id] = self.junction_scene[junction_id] + [scene_idx]
        
        if self.turning_yaw_diff[scene_idx] > 0:
            self.junction_turning_scene[junction_id]['Turning Right'] += [scene_idx]
        else:
            self.junction_turning_scene[junction_id]['Turning Left'] += [scene_idx]