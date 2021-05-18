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



