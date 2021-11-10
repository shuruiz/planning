# sampling module to generate guassian multi-variant samples 
# the output samples are local-coordinate based (assume the current point is origin, 
# and needed to be transformed to the sub-trajectory-level coordiantes, where the first point is the origin
# 

"""
example to use:

import sampling
S=sampling.sampling()
output= S.sample(sample_size = 1024)
print(output)
"""


import numpy as np 
import math 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import numpy as np 
import math
from math import cos, sin


class sampling():
    """
    The result of the class is to return a number of sampled points with shape [sqrt(sample_size), sqrt(sample_size), 3] 
    TO DO: transform back function to be implemented. 
    """

    def  __init__(self,point_coor=[0,0], heading=0.0, range_x=[-2,2.], range_y=[-2,2.], mean =[0.0, 0.6], cov=[[0.25, 0], [0, 0.25]]):
        """
        heading unit: rad
        point_coor: current investigated point

        """
        self.current_point=point_coor
        self.heading = heading
        self.range_x =range_x
        self.range_y=range_y
        self.sample_mean=mean
        self.sample_cov = cov

    def sample(self,sample_size=4096):
        """
        input: origin (x, y), should be the current point 
        heading: angle from previous point to the current point 
        range_x: radius of the ped reachable zone in left and right direction, maximum distance 
        range_y: radius of the ped reachable zone  in forward direction, maximum distance 

        ** reachable zone is assumed to be a half-circle, but can be oval if take different x, y values

        output: returns a [sample_size, 3] shape numpy array

        """ 
        x= np.mgrid[self.range_x[0]:self.range_x[1]:.01]
        y = []
        X=[]
        radius  =abs(self.range_x[1])
        for ele in x : 
            border = math.sqrt(radius**2-ele**2)
            y_x =np.mgrid[0:border:.01]    
            y_x = list(y_x)    
            X = X+ ([ele]*len(y_x))
            y = y+ y_x
        X=np.array(X)
        y=np.array(y)
        pos = np.dstack((X,y))[0]
        rv = multivariate_normal(self.sample_mean, self.sample_cov)
        prob= rv.pdf(pos)
        result=[]
        for xx, yy, p in zip(X, y, prob):
            result.append([xx, yy, p])
        result = np.array(result)
        results = result[np.random.choice(result.shape[0], sample_size, replace=False)]
        # idx = np.random.randint(low=0, high=len(result)-1, size=sample_size)
        # results = np.array(result[idx,:])

        on_x = np.array([[x] for x in results[:,0]])
        on_y = np.array([[y] for y in results[:,1]])
        on_p = np.array([[p] for p in results[:,2]])
        coordinates = np.hstack((on_x,on_y))

        rotated = self.rotate(coordinates).T

        transformed = self.transform_to_current_origin(rotated)  

        output = np.hstack((transformed, on_p))
        root = math.sqrt(sample_size)
        output=np.reshape(output, (int(root), int(root), 3))
        return output

    def rotate(self, coordinates):
        """
        cooridnates should be in [size, 2] shape, numpy 
        """
        theta =self.heading
        rotate_matrix = np.array([[cos(theta), sin(theta)],[-sin(theta), cos(theta)]])
        rotated = np.matmul(rotate_matrix, coordinates.T)
        return(rotated)


    def transform_to_current_origin(self, coordinates):
        """
        to do: transform the sampled points from local to original local coordinates

        the returned output should be a (sample size, 2) shape numpy array , i.e., [[x,y], [x',y'], ...]
        please see line 88 to call this function 

        """
        for point in coordinates:
            point += self.current_point
        return coordinates
        


