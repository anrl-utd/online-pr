import numpy as np
from online_environment import Environment

def eval_obj(actions, G):
    """ Returns the objective function (cumultive utility) of a sequence of actions

    :param actions: list of actions
    :G: networkx graph with params (util, demand) for each node
    :returns: cumulitive utility from taking the given action sequence in the graph
    """
    env = Environment(G)
    while env.
