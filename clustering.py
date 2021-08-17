import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import glob
import os
from tslearn.clustering import TimeSeriesKMeans
import copy as cp
import random
import pickle
import multiprocessing
import time
from collections import Counter

def clustering(feature, k, metrics="dtw"):

    n_jobs=multiprocessing.cpu_count()
    print("start training clustering on",n_jobs, "cpus")
    start_time = time.time()
    km_dba = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=30, max_iter_barycenter=5, n_jobs=-1, verbose=False, \
                                      random_state=np.random.randint(low=0,high=20,size=2)[0]).fit(feature)
    print("---used  %s seconds ---" % (time.time() - start_time))
    return km_dba


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("saved_centers"):
        os.makedirs("saved_centers")

    feature = np.load("data/Vehicle_trajectory.npy", allow_pickle=True)
    print(feature.shape)

    metric="dtw"
    n_initial = 10 # how many times to re-initial the run
    CENTERS=[]
    LABELS=[]
    k_min = 6
    k_max= 12
    elbow = []
    best_in_a_k=[]
    for k in range(k_min, k_max+1):
        print("runing number of clusters:", k)
        for i in range(0,n_initial):
            print("running re-initialization:", i, "out of", n_initial)
            km_dba = clustering(feature, k, metrics=metric)
            CENTERS.append(km_dba.cluster_centers_)
            LABELS.append(km_dba.labels_)
            best_in_a_k.append(km_dba.inertia_)
            pickle.dump(km_dba, open("saved_centers/cluster_model_k"+str(k)+"_i"+str(i)+".pkl", 'wb'))
        elbow.append(min(best_in_a_k))
    pickle.dump(elbow, open("elbow_list.pkl", 'wb'))
        
    plt.plot(range(k_min, k_max+1), elbow)
    plt.xlabel('Number of centroids')
    plt.ylabel('Elbow score')
    plt.title("Elbow score on vehicle features")
    plt.savefig("Elbow plot")
    plt.show
       