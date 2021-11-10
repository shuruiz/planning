import numpy as np
import gym
# from lib.gridworld import GridworldEnv

def reset(nA):
    POLICY = {}
    pie = np.ones(nA) * 1/nA
    for s,_ in trans_mat.items():
        POLICY[s] = pie
    V_init = np.zeros(len(trans_mat))
    return V_init, POLICY

def _onestep_q(V, s, a, trans_mat, gamma=1):
    q = 0
    for item in trans_mat[s][a]:
        p, s_next, rew, done = item
        q += p * (rew + gamma * V[s_next])
    return q

def _onestep_v(V, s, policy, trans_mat, gamma=1):
    v = 0
    for a,_ in trans_mat[s].items():
        q = _onestep_q(V, s, a, trans_mat, gamma)
        v += policy[s][a] * q
    delta = np.abs(v-V[s])
    return delta, v

def policy_eval(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    V = V_init
    U = V_init.copy()
    while True:
        delta = 0
        for s,_ in trans_mat.items():
            dd,v = _onestep_v(V, s, policy, trans_mat, gamma)
            delta = max(delta, dd)
            if inplace:
                V[s] = v
            else:
                U[s] = v
        if not inplace:
            V = U
        if delta < theta:
            return V

def policy_improve(V, s, trans_mat, gamma=1):
    q = {}
    for a in trans_mat[s].keys():
        q[a] = -np.inf
    for a,_ in trans_mat[s].items():
        q[a] = _onestep_q(V, s, a, trans_mat, gamma)
    max_q = max(q.values())
    count_max = 0
    for a,val in q.items():
        if q[a]==max_q:
            count_max += 1
    aa = {}
    for a in q.keys():
        aa[a] = 1/count_max if q[a] == max_q else 0
    return aa, q

def policy_iter(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    is_stable = False
    while not is_stable:
        is_stable = True
        V = policy_eval(trans_mat, V_init, policy, theta, gamma, inplace=inplace)
        for s,_ in trans_mat.items():
            aa, _ = policy_improve(V, s, trans_mat, gamma)
            if not np.array_equal(policy[s], aa):
                policy[s] = aa
                is_stable = False
    return V, policy

def value_iter(trans_mat, V_init, theta, gamma=1, inplace=True):
    V = V_init
    U = V_init.copy()
    next_iter = True
    policy = {}
    while next_iter:
        delta = 0
        for s,_ in trans_mat.items():
            aa, q = policy_improve(V, s, trans_mat, gamma)
            delta = max(delta, np.abs(V[s] - max(q.values())))
            if inplace:
                V[s] = max(q.values())
            else:
                U[s] = max(q.values())
        if not inplace:
            V = U
        if delta < theta:
            next_iter = False
    for s,_ in trans_mat.items():
        policy[s] = np.zeros(len(trans_mat[s]))
        aa, _ = policy_improve(V, s, trans_mat, gamma)
        policy[s] = aa
    return V, policy

def print_pol(policy,nrow,nA,a_list='nesw'):
    ncol = len(policy) // nrow
    for idx in range(nrow):
        for jdx in range(ncol):
            output = ''
            index = idx*nrow+jdx
            for kdx in range(nA):
                if policy[index][kdx] != 0:
                    output += a_list[kdx]
            print('{0:^{1}}'.format(output,len(a_list)+2),end='')
        print('\n')

if __name__ == '__main__':
    env = GridworldEnv(slip=0.2, episodic=False)
    trans_mat = env.P
    nA = env.nA
    np.set_printoptions(precision=2)
    inplace = False

    V_init, POLICY = reset(nA)
    V = policy_eval(trans_mat, V_init, POLICY, theta = 0.0001, gamma=0.9, inplace=inplace)
    print('uniformly random policy evaluation:')
    print(V.reshape(5,-1))

    V_init, POLICY = reset(nA)
    V, policy = policy_iter(trans_mat, V_init, POLICY, theta = 0.0001, gamma = 0.9, inplace=inplace)
    print('optimal value function after policy iteration')
    print(V.reshape(5,-1))
    print('optimal policy after policy iteration')
    print_pol(policy,5,4)

    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9, inplace=inplace)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    print('optimal policy after value iteration')
    print_pol(policy,5,4)

    env = GridworldEnv(slip=0.2, episodic=True)
    trans_mat = env.P
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 1., inplace=inplace)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9, inplace=inplace)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.8, inplace=inplace)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))

    # inplace = False
    #
    # V_init, POLICY = reset(nA)
    # V1 = policy_eval(trans_mat, V_init, POLICY, theta = 0.0001, gamma=0.9, inplace=inplace)
    # print(np.array_equal(V,V1))
    #
    # V_init, POLICY = reset(nA)
    # V1, policy1 = policy_iter(trans_mat, V_init, POLICY, theta = 0.0001, gamma = 0.9, inplace=inplace)
    # print(np.array_equal(V,V1))
    # print(np.array_equal(policy,policy1))
    #
    # V_init, POLICY = reset(nA)
    # V1, policy1 = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9, inplace=inplace)
    # print(np.array_equal(V,V1))
    # print(np.array_equal(policy,policy1))
    #
    # env = GridworldEnv(slip=0.2, episodic=True)
    # trans_mat = env.P
    # V_init, POLICY = reset(nA)
    # V1, policy1 = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 1., inplace=inplace)
    # print(np.array_equal(V,V1))
    # print(np.array_equal(policy,policy1))
    #
    # V_init, POLICY = reset(nA)
    # V1, policy1 = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9, inplace=inplace)
    # print(np.array_equal(V,V1))
    # print(np.array_equal(policy,policy1))
    #
    # V_init, POLICY = reset(nA)
    # V1, policy1 = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.8, inplace=inplace)
    # print(np.array_equal(V,V1))
    # print(np.array_equal(policy,policy1))
