import keras.backend as K
import numpy as np 


class EpsGreedyQPolicy():
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        n_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(q_values)
        return action

class GreedyQPolicy():
    """Implement the greedy policy
    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class DQN():
	def __init__(self,model, policy=None, test_policy=None):
		self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()
    def reset_states(self):
    	pass 
