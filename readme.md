# Update (06.19.2021)

Extracted agents' state (history translation, future translation, history yaw, future yaw) at 2 Hz and 5-second period for both history and future tags. The structure of stored data follows:

​	Junction_name -> Scene -> Frame -> Label -> Track_id -> {history_translation, future_translation, history_yaw, future_yaw}

Junction_name: 8KfB, lane_merge

Scene: the scene indexes of the dataset

Frame: the global frame indexes of the dataset, starting at the first frame of the scene and ending up with the last frame of the scene, at 5-frame step. Example: [100,118] ---> [100, 105, 110, 115, 118]

Label: ['Not_Set', 'Unknown', 'Dont_Care', 'Car', 'Van', 'Tram', 'Bus', 'Truck', 'Emergency Vehicle', 'Other Vehicle', 'Bicycle', 'Motorcycle', 'Cyclist', 'Motorcyclist', 'Pedestrian', 'Animal', 'AV_Dont_Care']

Track_id: the unique track_id of each agent. The track_ids are found at the current frame

State: the history and future arrays are in the real-world sequence.

Notation: The state corresponding to the current frame is the last element of history data;

​			   At the two sides of a scene, the history or future data are filled with None.

# Update (06.10.2021)



Extracted AV's state (translation, velocity, acceleration, yaw) and plotted them given target scene index. The following figure is one scene happened at the junction "8KfB", where there are pedestrians near by.

<img src="D:\GitHub\Clone\planning\ruixuan\target_scene_index\8KfB\AV_state.png" alt="AV_state" style="zoom:50%;" />

# Update (06.07.2021)

Since the junction "sGK1" have 4-all-way-stop signs, it is not qualified for the current test environment. Instead, two other locations are picked. The basic criteria is that the ego vehicle makes a turn at the junction and the junction is not fully controlled by the stop sign. The following two figures show their geographical configurations:

<img src="D:\GitHub\Clone\planning\ruixuan\target_scene_index\8KfB\8KfB.png" alt="8KfB" style="zoom:30%;" />

<img src="D:\GitHub\Clone\planning\ruixuan\target_scene_index\lane_merge\lane_merge.png" alt="lane_merge" style="zoom:30%;" />

Again, the scene index information is stored in \target_scene_index\location. In each sub folder, there are five files: location.png shows the global location where the scenes happened; 8KfB.png (or lane_merge.png) shows the configuration; example_scene.png shows one static plot where the trajectories of ego vehicle, agent vehicle and pedestrians can be found. scene_index(name).npy is the whole index list of the scenes; scene_index(name)_Pedestrian.npy is the scene index list containing the pedestrian label.

# Pipeline of scenario filtering procedure

## 1. Objective

​	Find the scene index corresponding to each intersection (junction) and notify each one's maneuver (right turn/ left turn/ straight) .

### 1.1 Data source

​	The raw data comes from train dataset (not train_full dataset) containing 16265 scenes and semantic map comes from the provided map file.

## 2. Outline of the scene filtering algorithm

​	1) From the provided semantic map, lane info and junction info are extracted into two dictionaries, "Lane" and "Junction"; with these two dictionaries, another dictionary depicting the connection of lanes and junctions, "Junction_Lane" is generated. 

​	2) Applying step 1) for every scene in train dataset to obtain a list "turning_scenes" containing the indexes of all probable scenes as "turning_frames" containing all corresponding frames from the "turning_scenes".  Preliminarily processed each scene based on the AV's yaw at the starting frame $\text{yaw}_{start}$ and the ending frame $\text{yaw}_{end}$. A subjective threshold 1.25 in rad is defined and if for one scene the absolute difference $|\text{yaw}_{start}-\text{yaw}_{end}| > \text{yaw}_{th}$, this scene would be expected to occur a tuning behavior.

​	3) For each scene in the "turning_scene" list, do the followings:

​		a) Using Bayesian changepoint detection to find out the frame interval where the AV is actually making a turn by working on the yaw profile of the whole scene (248 frames);

​		b)  For every three frame in the obtained turning interval, find out which lane that the AV is currently in and store the passing lanes as a set;

​		c) For each scene and the passing lane info during making a turn, iterate over the "Junction_Lane" dictionary to find which junction contains at least one shared lane of the passing lane info;

​		d) If there exists a junction, store it as the key value for this scene index as well as its turning orientation;

## 3. File Introduction

​	1) Raw data

​		There should be three files in order to load the data and process the map information: train.zarr, semantic_map.pb, visualisation_config.yaml.

​	2) utils folder

​		In this folder, it contains the MATLAB scripts to find the changepoints of yaw profile. I call these files in Python environment and if you may need to install MATLAB_Python API first to run the ipynb file.

​	3) scenario_intersection.ipynb

​		Since I prefer using jupyter notebook at the beginning of a coding project, this ipynb file carries most processing functionalities. 

​		a) In the first section, it reads the raw data from the data file and define the Map_Api;

​		b) In the second section, it defines a class called Scene, where all the functions are defined inside. It takes zarr_data and map_api as its input.

​		c) In the third section, it shows the main workflow. Firstly, we declare a Scene type object. Secondly, we generate the map information. So far these two corresponds to step 1). Thirdly, we do the preliminary processing to find the scenes which are probable to be turning cases, and it corresponds to step 2). Next, we iterate over all "turning_scene", which corresponds to step 3).







