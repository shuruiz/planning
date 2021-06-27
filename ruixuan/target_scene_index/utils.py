import numpy as np
from operator import itemgetter

data = np.load("D:\\GitHub\\Clone\\planning\\ruixuan\\target_scene_index\\8KfB\\Agent_state.npy", allow_pickle=True).item()

target_label = ['Pedestrian', 'Car', 'Van', 'Bus', 'Truck', 'EV', 'OV', 'Bicycle', 'Motorcycle']


class ParseAgentState(object):

    def __init__(self, data):
        self.data = data

    def agent_parse(self, scene, target_label):

        parsed_agent_state = {}
        scene_data = self.data[scene]

        for checking_frame in scene_data.keys():
            frame_agents = {}
            target_agents = itemgetter(*target_label)(scene_data[checking_frame])
            for type_agents in target_agents:
                if len(type_agents) > 0:
                    frame_agents.update(type_agents)

            parsed_agent_state[checking_frame] = frame_agents

        return parsed_agent_state


if __name__ == '__main__':

    Agent = ParseAgentState(data)
    agent_state = Agent.agent_parse(82, target_label)