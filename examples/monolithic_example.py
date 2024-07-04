from copy import copy
import warnings

import cvxpy as cp
import networkx as nx
import numpy as np

import graphvx_core as gvx


def main():
    digraph = nx.DiGraph(nx.circulant_graph(32, [1]))
    # digraph = nx.DiGraph(nx.star_graph(16))
    # digraph = nx.DiGraph(nx.barbell_graph(8, 8))
    # digraph = nx.DiGraph(nx.wheel_graph(16))

    adj = copy(nx.to_numpy_array(digraph))
    print(adj)
    adj[0, 2] = 0
    digraph = nx.DiGraph(adj)

    n_max = adj.shape[0]
    nodes = range(n_max)

    big_n = 3

    def node_f(i):
        x = cp.Variable((big_n, big_n), name='x_{}'.format(i), PSD=True)
        # x = cvxpy.Variable((big_n,), name='x_{}'.format(i))
        variables.append(x)
        tgt = np.eye(big_n) * i
        # tgt = np.ones((big_n,)) * i
        # tgt = np.array([1,2,3]) * i
        obj = cp.sum_squares(x - tgt)
        constr = [0 <= x, x <= n_max]
        id = i
        return obj, constr, [x], id

    def edge_f(edge, vars_i, vars_j):
        obj = 1. * cp.sum_squares(vars_i[0] * edge[0] - vars_j[0])
        constr = [cp.abs(vars_i[0] - vars_j[0]) <= 0.5]

        return obj, constr

    equiv_adj, equiv_edge_f = gvx.graph_converter.to_undirected_edge_f(edge_f, adj)
    equiv_digraph = nx.DiGraph(equiv_adj)

    variables = []
    g = gvx.GraphVX(equiv_digraph, node_f, equiv_edge_f)
    optimizer = gvx.CvxpyMonolithicSolver(g)
    optimizer.solve(m=cp.Minimize, verbose=False, solver=cp.CLARABEL, settings={"max_iter": 10000})

    for i, var in enumerate(variables):
        print(i)
        print(var.value)
        print(var.name())
    print(len(variables))
    print(g.value)


if __name__ == "__main__":
    main()
