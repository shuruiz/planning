import os
import pandas as pd 
import numpy as np
import math
import sampling


""" 
each sub_trajectory should be converted to local coordinate first

the script uses converted sub_trajectries as input, all long trajectories are split into 
sub_trajectories and stored in a folder before calling this script,
"""
def extractDataframes(txt):
	"""
	this function takes a txt file with the following characteristics:
	The first row contains all the frame numbers

	The second row contains all the pedestrian IDs
	
	The third row contains all the y-coordinates
	
	The fourth row contains all the x-coordinates
	This function separates the txt files into different data frames based on the 
	pedestrian ID
	"""
	data_frame_list = []
	ID = -1
	with open(txt, 'r') as file:
		loop = True
		x_coor = []
		y_coor = []
		while loop:
			# read line
			line = file.readline().split()
			if len(line) >= 3:
				# check for ID change and create dataframe if necessary
				if ID != int(float(line[1])):
					ID = int(float(line[1]))
					if len(x_coor) >= 1:
						df = pd.DataFrame({"x'": x_coor, # make dataframe
										   "y'": y_coor})
						data_frame_list.append(df) # add dataframe to list
					x_coor = []
					y_coor = []
				# attach x and y coordinates
				x_coor.append(np.float64(line[3]))
				y_coor.append(np.float64(line[2]))
			else:
				loop = False
				df = pd.DataFrame({"x'": x_coor, # make dataframe
										   "y'": y_coor})
				data_frame_list.append(df) # add dataframe to list
				break
	return data_frame_list # return list of dataframes

# here we define some global variables that we might change into given inputs later

def find_rotation_angle(A):
	"""
	this function finds the angle needed to rotate the coordinate system
	counterclowckwise so that the y axis of the new system matches the 
	direction of vector A

	"""
	mag = math.sqrt(A[0]*A[0]+A[1]*A[1])
	if mag == 0:
		angle = 0
	# this line will find the angle between the vector and the +y axis using 
	# using the dot product function
	else:
		angle = math.acos(A[1]/mag)                         
	# now since the rotation happens counterclockwise, we need to figure out
	# which direction does the vector point to. If the x coordinate is positive
	# we need 2pi - angle in order to rotate it properly
	if A[0] < 0:
		angle = 2*math.pi - angle
	return angle
def load_data(trajectory_path = None, feature_sec=10,label_sec=10,newDataFrequency=2, feature_frames = None, label_frames = None):
	"""
	INPUT:
		trajectory_path: string containing the directory in which the function will iterate the csv files
		feature_sec: int containing how many seconds the feature is going to last
		label_sec: int containing how many seconds the label is going to last
		newDataFrequency: int containing the new data frequency of the files
	for each sub-trajectry
		1 for each point in the sub-trajectry, compute its heading, do a sampling and return a [sqrt(sample_size), sqrt(sample_size), 3] 

		 2. record and stack those samples together, in the following way:  
			 the first feature_sec seconds data (samples of first 20 points) is stored as features
			 the last label_sec seconds data (only record the coordinate value) is stored as labels

			for each sub-trajectry the shape of the output should be:  
			feature [20,sqrt(sample_size), sqrt(sample_size), 3], 
			label   [20, 2]e

	II: for all sub-trajectires, do the same as I, and stack all features together ,and all labels together, 
	If there are 200 sub-trajectires, the final output of this function should be two numpy arrays : (assuming you are using the default sample size I set)

	feature [200, 20,64,64, 3], 
	label   [200, 20,2],
	feature_position

	"""
	# if you want to use frames insted of seconds, this if statement will switch the parameters so that the function will work
	# correctly for that amount of frames
	if feature_frames is not None and label_frames is not None:
		feature_sec = feature_frames
		label_sec = label_frames
		newDataFrequency = 1
	# assume the sub-trajectories are stored under folder sub_trajectories and has name trip1.csv, trip2.csv and etc. 
	filenames = os.listdir(trajectory_path) # return a list, with filepath of each sub-trajectry
	sampleSize = 16*16

	feature = []
	label = []
	feature_position = []
	for i in range(len(filenames)):
		if filenames[i][-3:] == "csv":
			print(filenames[i])
			df = pd.read_csv(trajectory_path+'/'+filenames[i])  # read a sub-trajectires in to pandas dataframe 
			#discard dataframe if it's not long enough
			if len(df.index) > (feature_sec*newDataFrequency+label_sec*newDataFrequency)+1:
				continue
			# get the two coordinate columns and put them in an organized list called coor_list
			coor_list = np.column_stack((df["x'"],df["y'"]))
			feature_list = []
			label_list = []
			feature_position_list = []
			count = 0
			point_heading_changed = False
			#iterate through coordinate list
			for point in coor_list:
				if count == 0: # first point used as origin
					count += 1
					pr_point = point
					heading_vector = np.array([0,0])
					continue
				
				# for each point in the data get the heading,
				# if the heading is going to be [0,0] then the person hasn't moved and the 
				# heading vector can't change
				if not (point == pr_point).all():
					heading_vector = point-pr_point
					point_heading_changed = True
				else:
					point_heading_changed = False
				pr_point = point 
					
				# from counts 1-feature_sec it appends to the feature list and feature position, 
				# after that it's on the label list
				if count <= feature_sec*newDataFrequency:
					if point_heading_changed or count == 1:
						S = sampling.sampling(point,
										  find_rotation_angle(heading_vector))
						output = S.sample(sampleSize)
					# add sample to feature_list
					feature_list.append(output)
					# add position to feature_position_list
					feature_position_list.append(point)
				else:
					label_list.append(point)
				count += 1
	
				if count> (feature_sec+label_sec)*newDataFrequency:
					break
				
			# stack feature and label lists 
			feature.append(feature_list)
			label.append(label_list)
			feature_position.append(feature_position_list)
		
		elif filenames[i][-3:] == "txt":
			"""
			if file is a txt it contains multiple subtrajectories and we need
			to iterate through the list treating each ID as one sub_trajectory
			keep in mind that for txt files the frequency is 1 HZ
			"""
			print(filenames[i])
			data_frame_list = extractDataframes(trajectory_path+'/'+filenames[i])

			for df in data_frame_list:
				coor_list = np.column_stack((df["x'"],df["y'"]))
				origin = coor_list[0] #origin to transform back
				coor_list=coor_list[1:]

				if len(coor_list)< (feature_sec+label_sec):
					continue

				right_border=len(coor_list)-feature_sec-label_sec
				for i in range(0,right_border):
					coor_list_new = coor_list[i:i+feature_sec+label_sec]
					feature_list = []
					label_list = []
					feature_position_list = []
					count = 0


					#iterate through coordinate list

					for point in coor_list_new:
						################# useless block below#############
						if count == 0:
							count += 1
							pr_point = point
							heading_vector = point - pr_point
							continue

						 ################# useless block below#############
						if not (point == pr_point).all():
							heading_vector = point-pr_point
						 
						
						if count < feature_sec:
							if count == 1:

								output=[0]

								feature_list.append(output)
								feature_position_list.append(pr_point)

							output=[0]
							# add sample to feature_list
							feature_list.append(output)
							# add position to feature_position_list
							feature_position_list.append(point)
						else:
							label_list.append(point)
						count += 1
						if count > feature_sec+label_sec:
							break
						pr_point = point
					# stack feature and label lists adn return them
					feature.append(feature_list)
					# label.append(label_list)
					label.append(label_list[0])
					feature_position.append(feature_position_list)
	feature = np.array(feature)
	label = np.array(label)
	feature_position = np.array(feature_position)
	print(label.shape)
	print(feature.shape)
	return feature, label, feature_position
#feature, label, feature_position = load_data(os.getcwd()+"/txtTrajectories",10, 10, 2)
#np.save("npyArrays/10s_txt_feature.npy", feature)
#np.save("npyArrays/10s_txt_label.npy", label)
#np.save("npyArrays/10s_txt_feature_position.npy", feature_position)
