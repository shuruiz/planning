import numpy as np
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, Conv1D
from keras.layers import LSTM, Reshape
from keras.layers import GRU
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf

# add pooling

def _build_model(n_actions):
    model_sub_in = Input(shape=(1, 9))
    # start x ,start y, goalx,goaly, posx, posy, v, a, theta
    model_sub_flatten = Flatten()(model_sub_in)
    model_sub_mid = Dense(128, activation='relu', name='layer_sub_mid')(model_sub_flatten)
    model_sub_out = Dense(256, activation='elu', name='layer_sub_out')(model_sub_mid)
    model_sub = Model(model_sub_in, model_sub_out)

    model_veh_in = Input(shape=(5, 10, 2))
    # history traj veh
    model_veh_d = Dense(1,  activation='relu', name='layer_veh_dx')(model_veh_in)
    veh_reshape = Reshape((-1,10))(model_veh_d)
    model_veh_lstm = LSTM(256)(veh_reshape)
    veh_reshape2 = Reshape((4,8,8))(model_veh_lstm)
    
    model_veh_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_veh_conv1')(veh_reshape2)
    model_veh_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_veh_conv2')(model_veh_conv1)
    model_veh_flatten = Flatten()(model_veh_conv2)
    model_veh_out = Dense(128,  activation='relu', name='layer_veh_out')(model_veh_flatten)
    model_veh = Model(model_veh_in, model_veh_out)
    
    model_ped_in = Input(shape=(3, 10, 2))
    # history traj ped
    model_ped_d= Dense(1,  activation='relu', name='layer_ped_dx')(model_ped_in)
    ped_reshape = Reshape((-1,10))(model_ped_d)
    model_ped_lstm = LSTM(256)(ped_reshape)
    ped_reshape2 = Reshape((4,8,8))(model_ped_lstm)
    
    model_ped_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_ped_conv1')(ped_reshape2)
    model_ped_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_ped_conv2')(model_ped_conv1)
    model_ped_flatten = Flatten()(model_ped_conv2)
    model_ped_out = Dense(128,  activation='elu', name='layer_ped_out')(model_ped_flatten)
    model_ped = Model(model_ped_in, model_ped_out)
    
    
    model_cyc_in = Input(shape=(3, 10, 2))
    # history traj cyc
    model_cyc_d= Dense(1,  activation='relu', name='layer_cyc_dx')(model_cyc_in)
    cyc_reshape = Reshape((-1,10))(model_cyc_d)
    model_cyc_lstm = LSTM(256)(cyc_reshape)
    cyc_reshape2 = Reshape((4,8,8))(model_cyc_lstm)
    
    model_cyc_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv1')(cyc_reshape2)
    model_cyc_conv2 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv2')(model_cyc_conv1)
    model_cyc_flatten = Flatten()(model_cyc_conv2)
    model_cyc_out = Dense(128,  activation='elu', name='layer_cyc_out')(model_cyc_flatten)
    model_cyc = Model(model_cyc_in, model_cyc_out)
    
    
    
    model_edge_in = Input(shape=(1, 11))
    # edges connected to subject
    model_edge_flatten = Flatten()(model_edge_in)
    model_edge_mid = Dense(128,  activation='relu', name='layer_edge_mid')(model_edge_flatten)
    model_edge_out = Dense(256,  activation='relu', name='layer_edge_output')(model_edge_mid)
    model_edge = Model(model_edge_in, model_edge_out)


    concatenated = concatenate([model_sub_out, model_veh_out,model_ped_out,model_cyc_out,model_edge_out])
    graph_out = Dense(512, activation='softmax', name='graph_out_layer')(concatenated)
    
    # define the deep RL model below.
    rl0 = Dense(512,  activation='relu', name='layer_rl_0')(graph_out)
    rl1 = Dense(1024,  activation='relu', name='layer_rl_1')(rl0)
    rl2 = Dense(1024,  activation='relu', name='layer_rl_2')(rl1)
    out = Dense(n_actions,  activation='relu', name='layer_q_values')(rl2)
    
    # action space 
#     print(len(np.arange(-3,3,0.1)))
#     print(len(np.arange(0,360,5)))
    
    merged_model = Model([model_sub_in, model_veh_in,model_ped_in,model_cyc_in,model_edge_in], out)
    return merged_model

def _build_simple_model(n_actions):
    model_sub_in = Input(shape=(1, 9))
    # start x ,start y, goalx,goaly, posx, posy, v, a, theta
    model_sub_flatten = Flatten()(model_sub_in)
    # model_sub_mid = Dense(128, activation='relu', name='layer_sub_mid')(model_sub_flatten)
    model_sub_out = Dense(256, activation='elu', name='layer_sub_out')(model_sub_flatten)
    model_sub = Model(model_sub_in, model_sub_out)

    model_veh_in = Input(shape=(5, 10, 2))
    # history traj veh
    model_veh_d = Dense(1,  activation='relu', name='layer_veh_dx')(model_veh_in)
    veh_reshape = Reshape((-1,10))(model_veh_d)
    model_veh_lstm = LSTM(128)(veh_reshape)
    veh_reshape2 = Reshape((2,8,8))(model_veh_lstm)
    
    model_veh_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_veh_conv1')(veh_reshape2)
    # model_veh_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_veh_conv2')(model_veh_conv1)
    model_veh_flatten = Flatten()(model_veh_conv1)
    model_veh_out = Dense(128,  activation='relu', name='layer_veh_out')(model_veh_flatten)
    model_veh = Model(model_veh_in, model_veh_out)
    
    model_ped_in = Input(shape=(3, 10, 2))
    # history traj ped
    model_ped_d= Dense(1,  activation='relu', name='layer_ped_dx')(model_ped_in)
    ped_reshape = Reshape((-1,10))(model_ped_d)
    model_ped_lstm = LSTM(128)(ped_reshape)
    ped_reshape2 = Reshape((2,8,8))(model_ped_lstm)
    
    model_ped_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_ped_conv1')(ped_reshape2)
    # model_ped_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_ped_conv2')(model_ped_conv1)
    model_ped_flatten = Flatten()(model_ped_conv1)
    model_ped_out = Dense(128,  activation='elu', name='layer_ped_out')(model_ped_flatten)
    model_ped = Model(model_ped_in, model_ped_out)
    
    
    model_cyc_in = Input(shape=(3, 10, 2))
    # history traj cyc
    model_cyc_d= Dense(1,  activation='relu', name='layer_cyc_dx')(model_cyc_in)
    cyc_reshape = Reshape((-1,10))(model_cyc_d)
    model_cyc_lstm = LSTM(128)(cyc_reshape)
    cyc_reshape2 = Reshape((2,8,8))(model_cyc_lstm)
    
    model_cyc_conv1 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv1')(cyc_reshape2)
    # model_cyc_conv2 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv2')(model_cyc_conv1)
    model_cyc_flatten = Flatten()(model_cyc_conv1)
    model_cyc_out = Dense(128,  activation='elu', name='layer_cyc_out')(model_cyc_flatten)
    model_cyc = Model(model_cyc_in, model_cyc_out)
    
    
    
    model_edge_in = Input(shape=(1, 11))
    # edges connected to subject
    model_edge_flatten = Flatten()(model_edge_in)
    # model_edge_mid = Dense(128,  activation='relu', name='layer_edge_mid')(model_edge_flatten)
    model_edge_out = Dense(256,  activation='relu', name='layer_edge_output')(model_edge_flatten)
    model_edge = Model(model_edge_in, model_edge_out)


    concatenated = concatenate([model_sub_out, model_veh_out,model_ped_out,model_cyc_out,model_edge_out])
    graph_out = Dense(512, activation='softmax', name='graph_out_layer')(concatenated)
    
    # define the deep RL model below.
    rl0 = Dense(512,  activation='relu', name='layer_rl_0')(graph_out)
    # rl1 = Dense(1024,  activation='relu', name='layer_rl_1')(rl0)
    # rl2 = Dense(1024,  activation='relu', name='layer_rl_2')(rl1)
    out = Dense(n_actions,  activation='relu', name='layer_q_values')(rl0)
    
    # action space 
#     print(len(np.arange(-3,3,0.1)))
#     print(len(np.arange(0,360,5)))
    
    merged_model = Model([model_sub_in, model_veh_in,model_ped_in,model_cyc_in,model_edge_in], out)
    return merged_model


def _build_simple_model2(n_actions):
    model_sub_in = Input(shape=(1, 9))
    # start x ,start y, goalx,goaly, posx, posy, v, a, theta
    model_sub_flatten = Flatten()(model_sub_in)
    # model_sub_mid = Dense(128, activation='relu', name='layer_sub_mid')(model_sub_flatten)
    model_sub_out = Dense(256, activation='elu', name='layer_sub_out')(model_sub_flatten)
    model_sub = Model(model_sub_in, model_sub_out)

    model_veh_in = Input(shape=(5, 10, 2))
    # history traj veh
    model_veh_d = Dense(1,  activation='relu', name='layer_veh_dx')(model_veh_in)
    veh_reshape = Reshape((-1,10))(model_veh_d)
    model_veh_lstm = LSTM(64)(veh_reshape)
    veh_reshape2 = Reshape((4,4,4))(model_veh_lstm)
    
    model_veh_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_veh_conv1')(veh_reshape2)
    # model_veh_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_veh_conv2')(model_veh_conv1)
    model_veh_flatten = Flatten()(model_veh_conv1)
    model_veh_out = Dense(64,  activation='relu', name='layer_veh_out')(model_veh_flatten)
    model_veh = Model(model_veh_in, model_veh_out)
    
    model_ped_in = Input(shape=(3, 10, 2))
    # history traj ped
    model_ped_d= Dense(1,  activation='relu', name='layer_ped_dx')(model_ped_in)
    ped_reshape = Reshape((-1,10))(model_ped_d)
    model_ped_lstm = LSTM(64)(ped_reshape)
    ped_reshape2 = Reshape((4,4,4))(model_ped_lstm)
    
    model_ped_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_ped_conv1')(ped_reshape2)
    # model_ped_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_ped_conv2')(model_ped_conv1)
    model_ped_flatten = Flatten()(model_ped_conv1)
    model_ped_out = Dense(64,  activation='elu', name='layer_ped_out')(model_ped_flatten)
    model_ped = Model(model_ped_in, model_ped_out)
    
    
    model_cyc_in = Input(shape=(3, 10, 2))
    # history traj cyc
    model_cyc_d= Dense(1,  activation='relu', name='layer_cyc_dx')(model_cyc_in)
    cyc_reshape = Reshape((-1,10))(model_cyc_d)
    model_cyc_lstm = LSTM(64)(cyc_reshape)
    cyc_reshape2 = Reshape((4,4,4))(model_cyc_lstm)
    
    model_cyc_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_cyc_conv1')(cyc_reshape2)
    # model_cyc_conv2 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv2')(model_cyc_conv1)
    model_cyc_flatten = Flatten()(model_cyc_conv1)
    model_cyc_out = Dense(64,  activation='elu', name='layer_cyc_out')(model_cyc_flatten)
    model_cyc = Model(model_cyc_in, model_cyc_out)
    
    
    
    model_edge_in = Input(shape=(1, 11))
    # edges connected to subject
    model_edge_flatten = Flatten()(model_edge_in)
    # model_edge_mid = Dense(128,  activation='relu', name='layer_edge_mid')(model_edge_flatten)
    model_edge_out = Dense(256,  activation='relu', name='layer_edge_output')(model_edge_flatten)
    model_edge = Model(model_edge_in, model_edge_out)


    concatenated = concatenate([model_sub_out, model_veh_out,model_ped_out,model_cyc_out,model_edge_out])
    graph_out = Dense(256, activation='softmax', name='graph_out_layer')(concatenated)
    
    # define the deep RL model below.
    rl0 = Dense(256,  activation='relu', name='layer_rl_0')(graph_out)
    # rl1 = Dense(1024,  activation='relu', name='layer_rl_1')(rl0)
    # rl2 = Dense(1024,  activation='relu', name='layer_rl_2')(rl1)
    out = Dense(n_actions,  activation='relu', name='layer_q_values')(rl0)
    
    # action space 
#     print(len(np.arange(-3,3,0.1)))
#     print(len(np.arange(0,360,5)))
    
    merged_model = Model([model_sub_in, model_veh_in,model_ped_in,model_cyc_in,model_edge_in], out)
    return merged_model



def _build_reduced_model(n_actions):
    model_sub_in = Input(shape=(1, 9))
    # start x ,start y, goalx,goaly, posx, posy, v, a, theta
    model_sub_flatten = Flatten()(model_sub_in)
    # model_sub_mid = Dense(128, activation='relu', name='layer_sub_mid')(model_sub_flatten)
    model_sub_out = Dense(256, activation='elu', name='layer_sub_out')(model_sub_flatten)
    model_sub = Model(model_sub_in, model_sub_out)

    model_veh_in = Input(shape=(5, 10, 2))
    # history traj veh
    model_veh_d = Dense(1,  activation='relu', name='layer_veh_dx')(model_veh_in)
    veh_reshape = Reshape((-1,10))(model_veh_d)
    model_veh_lstm = LSTM(64)(veh_reshape)
    veh_reshape2 = Reshape((4,4,4))(model_veh_lstm)
    
    model_veh_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_veh_conv1')(veh_reshape2)
    # model_veh_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_veh_conv2')(model_veh_conv1)
    model_veh_flatten = Flatten()(model_veh_conv1)
    model_veh_out = Dense(4,  activation='relu', name='layer_veh_out')(model_veh_flatten)
    model_veh = Model(model_veh_in, model_veh_out)
    
    model_ped_in = Input(shape=(3, 10, 2))
    # history traj ped
    model_ped_d= Dense(1,  activation='relu', name='layer_ped_dx')(model_ped_in)
    ped_reshape = Reshape((-1,10))(model_ped_d)
    model_ped_lstm = LSTM(64)(ped_reshape)
    ped_reshape2 = Reshape((4,4,4))(model_ped_lstm)
    
    model_ped_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_ped_conv1')(ped_reshape2)
    # model_ped_conv2 = Conv2D(256,(2,2),  activation='relu', name='layer_ped_conv2')(model_ped_conv1)
    model_ped_flatten = Flatten()(model_ped_conv1)
    model_ped_out = Dense(4,  activation='elu', name='layer_ped_out')(model_ped_flatten)
    model_ped = Model(model_ped_in, model_ped_out)
    
    
    model_cyc_in = Input(shape=(3, 10, 2))
    # history traj cyc
    model_cyc_d= Dense(1,  activation='relu', name='layer_cyc_dx')(model_cyc_in)
    cyc_reshape = Reshape((-1,10))(model_cyc_d)
    model_cyc_lstm = LSTM(64)(cyc_reshape)
    cyc_reshape2 = Reshape((4,4,4))(model_cyc_lstm)
    
    model_cyc_conv1 = Conv2D(64,(2,2),  activation='relu', name='layer_cyc_conv1')(cyc_reshape2)
    # model_cyc_conv2 = Conv2D(128,(2,2),  activation='relu', name='layer_cyc_conv2')(model_cyc_conv1)
    model_cyc_flatten = Flatten()(model_cyc_conv1)
    model_cyc_out = Dense(4,  activation='elu', name='layer_cyc_out')(model_cyc_flatten)
    model_cyc = Model(model_cyc_in, model_cyc_out)
    
    
    
    model_edge_in = Input(shape=(1, 11))
    # edges connected to subject
    model_edge_flatten = Flatten()(model_edge_in)
    # model_edge_mid = Dense(128,  activation='relu', name='layer_edge_mid')(model_edge_flatten)
    model_edge_out = Dense(4,  activation='relu', name='layer_edge_output')(model_edge_flatten)
    model_edge = Model(model_edge_in, model_edge_out)


    concatenated = concatenate([model_sub_out, model_veh_out,model_ped_out,model_cyc_out,model_edge_out])
    graph_out = Dense(256, activation='softmax', name='graph_out_layer')(concatenated)
    
    # define the deep RL model below.
    rl0 = Dense(256,  activation='relu', name='layer_rl_0')(graph_out)
    # rl1 = Dense(1024,  activation='relu', name='layer_rl_1')(rl0)
    # rl2 = Dense(1024,  activation='relu', name='layer_rl_2')(rl1)
    out = Dense(n_actions,  activation='relu', name='layer_q_values')(rl0)
    
    # action space 
#     print(len(np.arange(-3,3,0.1)))
#     print(len(np.arange(0,360,5)))
    
    merged_model = Model([model_sub_in, model_veh_in,model_ped_in,model_cyc_in,model_edge_in], out)
    return merged_model
if __name__ =='__main__':
    # model = _build_simple_model(480)
    # model = _build_simple_model2(800)
    model = _build_reduced_model(800)
    model.summary()