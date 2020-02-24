import numpy as np
from graph_helper import r_tree, plot_graph
from online_environment import Environment
import networkx as nx
import time


def m_hinge(x, eps=0.01):
    return max((1/eps)*x + 1, 0)


def c_hinge(x):
    return max(-x, -1)


def n_hinge(x, eps=0.01):
    return max(-(1/eps)*x + 1, 0)


def m_hinge_vec(V):
    for idx in range(len(V)):
        V[idx] = m_hinge(V[idx])

    return V


def c_hinge_vec(V):
    print(V)
    for idx in range(len(V)):
        V[idx] = c_hinge(V[idx])

    return V


def allocation_filter_matrix(N, S):
    """ Construct the node allocation filter matrix (N x NS) """
    print("N={0}, S={1}".format(N,S))
    return np.tile(np.eye(N), (1, S))


def step_filter_matrix(N, S):
    """ Construct the step filter matrix (N x NS) """
    print("N={0}, S={1}".format(N, S))
    return np.kron(np.eye(S), np.ones((1, N)))


def get_adjacent_nodes(G, func_nodes):
    """
    Get nodes that are either functional or adjacent to functional in G

    :param G: networkx graph
    :param func_nodes: indices of functional nodes
    :return: 0-1 vector of functional nodes/adjacent to functional nodes
    """
    c = [0 for _ in range(G.number_of_nodes())]
    for node in G:
        for f_node in func_nodes:
            if node in G.neighbors(f_node):
                c[node] = 1
                break
    return np.array(c)


def eval_obj(actions, G, independent_nodes, resources):
    """
    Returns the objective function (cumulative utility) of a sequence of actions

    :param actions: list of actions
    :param G: networkx graph with params (util, demand) for each node
    :param independent_nodes: initially functional nodes in G
    :param resources: resources per turn
    :returns: cumulative utility from taking the given action sequence in the graph
    """
    num_nodes = G.number_of_nodes()
    env = Environment(G, independent_nodes, resources)
    utils = [0 for _ in range(num_nodes)]
    for key, val in nx.get_node_attributes(G, 'util').items():
        utils[key] = val

    reward = 0
    m = []
    c = []
    F = [np.array([-1 if idx in independent_nodes else 0 for idx in range(num_nodes)])]
    start_demand = np.array([env.start_demand[x] for x in range(num_nodes)])
    allocated = np.array([0 for _ in range(num_nodes)])

    for action_i in actions:
        state, r, done = env.step(action_i, debug=True)
        reward += r
        allocated += np.array(action_i)
        m_s = allocated - start_demand
        m.append([m_hinge(m_s[x]) for x in range(len(m_s))])
        # print("Functional nodes:", env.get_functional_nodes())
        print("F_i", F[-1], F[-1].shape)
        c_s_no_hinge = np.dot(F[-1], nx.to_numpy_matrix(G)).tolist()[0]
        c_s = [c_hinge(v) for v in c_s_no_hinge]
        c.append(c_s)
        print("c_s", c[-1])
        print(action_i)
        print("m_s", m_s)
        F.append(np.array([m[-1][idx] * c[-1][idx] for idx in range(len(c[-1]))]))
        if done:
            break

    obj = 0
    for step in range(len(F)):
        obj += sum([utils[idx] * F[step][idx] for idx in range(len(utils))])
    print('Objective function:', obj)

    return reward


def create_u_t(G, util, demand, r):
    N = len(util)
    S = int(np.ceil(np.sum(demand) / r))
    M_n = allocation_filter_matrix(N, S)
    M_w = step_filter_matrix(N, S)
    # initial functionality vector, always start recovery from node 0
    F_0 = np.zeros(N)
    F_0[0] = 1
    util[0] = 0
    demand[0] = 0

    adj_matrix = nx.to_numpy_matrix(G)

    def calc_obj(L):
        """ Use L to calculate the objective function. This will be passed into autograd """
        lam_1 = 1e7
        lam_2 = 1e7 
        lam_3 = 1e7 
        # do stuff
        obj = 0
        F_last = F_0 
        F_s = [F_0]
        for s in range(1, S+1):
            m_s = m_hinge_vec(np.matmul(M_n, L) - demand)
            print(F_last.T, '\nadj', adj_matrix, '\nres', np.matmul(F_last.T, adj_matrix))
            print('next\n')
            res = np.matmul(F_last.T, adj_matrix)
            c_s = c_hinge_vec(res.reshape((N,)))
            # element wise multiply
            F_last = np.multiply(m_s, c_s)
            F_s.append(F_last)
            obj += np.dot(util, F_last)

        # First constraint
        pi_1 = lam_1 * np.dot(np.ones(N), np.max(np.matmul(M_w, L) - np.full(N, r), 0))

        # Second constraint
        pi_2 = 0

        # Third constraint
        pi_3 = 0

        return obj + pi_1 + pi_2 + pi_3

    # Return a function which takes in L as an argument
    return calc_obj


def test_matrix_generation():
    print("testing allocation filter matrix")
    print(allocation_filter_matrix(2, 2))
    print(allocation_filter_matrix(3, 2))
    print(allocation_filter_matrix(4, 3))
    print("")

    print("testing step filter matrix")
    print(step_filter_matrix(2,2))
    print(step_filter_matrix(5,3))
    print(step_filter_matrix(4, 2))


def main():
    num_nodes = 7
    resources = 1
    G = r_tree(num_nodes)

    plot_graph(G, 0, 'environment_debug_graph.png')
    utils = [0 for _ in range(num_nodes)]
    for key, val in nx.get_node_attributes(G, 'util').items():
        utils[key] = val
    demand = [0 for _ in range(num_nodes)]
    for key, val in nx.get_node_attributes(G, 'demand').items():
        demand[key] = val

    N = len(utils)
    S = int(np.ceil(np.sum(demand) / resources))
    obj_func = create_u_t(G, utils, demand, resources)

    print(obj_func(np.zeros(N * S)))


if __name__=='__main__':
    main()
