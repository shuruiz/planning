import numpy as np 
import math


def crop(array, max_n_agent=10, is_cyc=False):
    """
    preapre for training
    """
    if len(array)>=max_n_agent:
        return np.array(array[:max_n_agent])
    if len(array)<max_n_agent:
        complement = np.zeros(shape=(max_n_agent-len(array), 20, 2))
        if len(array)==0:return complement
        if is_cyc:
            replica = math.ceil(max_n_agent/len(array))
            array=list(array)*replica
            return np.array(array[:max_n_agent])
        array = np.vstack((array, complement))
        return array
    else:
        return array
    
# utility functions
def get_distance(v1,v2):
    """
    get distance between points in v1 and v2
    """
    return np.linalg.norm(v1 - v2, axis=1)

def get_distance_pt(p1, p2):
    """
    get l2 distance between two point
    """
    return np.linalg.norm(p1 - p2)


def percent_left(node):
    """
    get l2 distance between two point
    """
    all_dist = get_distance_pt(node.start, node.goal)
    dist_left = get_distance_pt(node.pos, node.goal)

    # print("dist to goal", dist_left, 100* dist_left / all_dist)
    return dist_left
    # if all_dist==0:
    #     return 0
    # return 100* dist_left / all_dist

def get_socially_acceptable(veh):
    """
    greedy algorithm
    """
    result_veh=[]
    n = len(veh)
    x = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i,n):
#             print(i,j)
            distance = get_distance(veh[i], veh[j])
            x[i][j] = np.min(distance)
#             if np.min(distance)<4:
#                 break
    # add all that is distance> 4 with the first sample
    result_veh.append(veh[0])
    for i in range(1,n):
        if x[0][i]>4:
            result_veh.append(veh[i])
    return result_veh

def get_socially_acceptable_different_types(type1, type2):
    """
    greedy algorithm, type1 is the anchor type 
    """
    result=[]
    all_pass=True
    for type2_traj in type2:
        for type1_traj in type1:
            distance = get_distance(type1_traj, type2_traj)
            if min(distance)<4:
                all_pass=False
                break 
        if all_pass:
            result.append(type2_traj)
    return result

def socially_acceptance_check(veh, ped, cyc):
    #check socially acceptable trajectories
    veh_s = get_socially_acceptable(veh)
    ped_s = get_socially_acceptable(ped)
    cyc_s = get_socially_acceptable(cyc)
    
    # check constrains on ped_s
    ped_s = get_socially_acceptable_different_types(veh_s, ped_s)
    
    #check socially acceptable trajectories considering environment
    cyc_s = get_socially_acceptable_different_types(ped_s, cyc_s)
    cyc_s = get_socially_acceptable_different_types(veh_s, cyc_s)
#     print(len(veh_s), len(ped_s), len(cyc_s))
    return np.array(veh_s), np.array(ped_s), np.array(cyc_s)

# sampling strategies
def MCMC_sampling(final_veh, final_ped, final_cyc, max_n_veh=8, max_n_ped=5, max_n_cyc=6):
    """
    random sampling on all veh, ped, cyc trajectories separately to generate trajectories in a scene
    """
    print(final_veh.shape, final_ped.shape, final_cyc.shape)
    veh =final_veh[np.random.choice(final_veh.shape[0], max_n_veh, replace=False)]
    ped =final_ped[np.random.choice(final_ped.shape[0], max_n_ped, replace=False)]
    cyc =final_cyc[np.random.choice(final_cyc.shape[0], max_n_cyc, replace=False)]
    
    return veh, ped, cyc
#     print(veh.shape, ped.shape)

def Gibbs_sampling(max_scene, Pxy, Pxz, Pyzx, \
                   poolv, poolp, poolc,\
                   veh_model, ped_model, cyc_model, \
                   max_n_veh=10, max_n_ped=5, max_n_cyc=5, tasks=None):
    sampling_result={}
    
    # initialization
    veh_0, ped_0, cyc_0 = poolv[list(poolv.keys())[0]], poolp[list(poolp.keys())[0]], poolc[list(poolc.keys())[0]], 
    veh =veh_0[np.random.choice(veh_0.shape[0], max_n_veh, replace=False)]
    ped =ped_0[np.random.choice(ped_0.shape[0], max_n_ped, replace=False)]
    cyc =cyc_0[np.random.choice(cyc_0.shape[0], max_n_cyc, replace=False)]
    
    n_scene=0
    while n_scene < max_scene:
        scene_data={}
        veh_p = crop(veh)
        ped_p = crop(ped)
        cyc_p = crop(cyc,is_cyc=True)
        x = veh_model.predict([veh_p.flatten()])[0]
        y = ped_model.predict([ped_p.flatten()])[0]
        z = cyc_model.predict([cyc_p.flatten()])[0]
#         print(x, y, z, Pxy[x])
        
        pxy = Pxy[x]/np.linalg.norm(Pxz[x], ord=1)
        ped_category = np.random.choice(list(poolp.keys()), p=pxy)
        #sample trajectories
        ped_under_category = poolp[ped_category]
        ped =ped_under_category[np.random.choice(ped_under_category.shape[0], max_n_ped, replace=False)]
#         ped = poolp[ped_category][ped_idx]
#         print(ped.shape)
        
        pxz = Pxz[x]/np.linalg.norm(Pxz[x], ord=1)
        key_list= list(poolc.keys())
        if len(key_list)!= len(pxz):
            key_list=range(len(pxz))
        cyc_category = np.random.choice(key_list, p=pxz)
        cyc_under_category = poolc[cyc_category]
        cyc =cyc_under_category[np.random.choice(cyc_under_category.shape[0], max_n_cyc, replace=False)]
        
#         print(Pyzx.shape, Pyzx[y][z].shape)
        pyzx = Pyzx[y][z]/np.linalg.norm(Pyzx[y][z], ord=1)
        v_category = np.random.choice(list(poolv.keys()), p=pyzx)
        veh_under_category = poolv[v_category]
        veh =veh_under_category[np.random.choice(veh_under_category.shape[0], max_n_veh, replace=False)]
        
        


        veh, ped, cyc = socially_acceptance_check(veh, ped, cyc)
        # print(veh.shape)
        if len(veh)<=1 or len(ped)==0 or len(cyc)==0:
            continue

        if tasks is not None:
            task = np.array(tasks[np.random.choice(len(tasks), 1, replace=False)[0]])  # random select a task from task pool
            # task = np.array(tasks[0]) # only one task, going through


            task = np.expand_dims(task, axis=0)
            # print("task shape",task.shape)
            veh = np.vstack((task, veh))

        
        scene_data['ped']=np.round(ped, decimals=3)
        scene_data['cyc']=np.round(cyc, decimals=3)
        scene_data['veh']=np.round(veh,decimals=3)
        sampling_result[n_scene] = scene_data
        n_scene+=1
    return sampling_result

def extract_feature(ele):
    """
    extract feature for a single trajectory
    f = [x, y, a,v, delta_a, delta_v]
    """
    fe=[]
    for i in range(3, len(ele)):
        prev3, prev2, prev, curr = ele[i-3], ele[i-2], ele[i-1], ele[i]
        v = np.linalg.norm(curr - prev)/0.5
        v_prev = np.linalg.norm(prev - prev2)/0.5
        v_prev2 = np.linalg.norm(prev2 - prev3)/0.5
        a = (v-v_prev)/0.5
        a_prev = (v_prev-v_prev2)/0.5
        
        energy = v**2-v_prev**2
        force = a-a_prev
        f = [ele[i][0], ele[i][1], v, a, energy, force]
        fe.append(f)
    return np.array(fe)

def get_smoothness(traj):
    pass