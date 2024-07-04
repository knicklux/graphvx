from copy import copy

import cvxpy as cp
import numpy as np


def placeholder_edge_f(edge, vars_i, vars_j):
    return cp.abs(0), []


def to_underected_graph(adj):
    n = adj.shape[0]
    adj = copy(adj)
    removed_ji = np.zeros((n, n))
    dummy_ij = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if adj[i, j]:
                removed_ji[j, i] = adj[j, i]
                adj[j, i] = 0
    for i in range(n):
        for j in range(i, n):
            if adj[j, i]:
                dummy_ij[i, j] = 1
                adj[j, i] = 0
                removed_ji[j, i] = 1
    return adj + dummy_ij, removed_ji, dummy_ij


def augmented_edge_f(edge, vars_i, vars_j, edge_f, removed_ji, dummy_ij):
    i, j = edge

    obj = cp.square(0)
    constr = []
    # print("edge: {}, {}".format(i, j))

    if i < j:
        if dummy_ij[i, j]:
            # print("dummy")
            obj_, constr_ = placeholder_edge_f(edge, vars_i, vars_j)
            obj += obj_
            constr += constr_
        else:
            # print("(i, j) present")
            obj_, constr_ = edge_f(edge, vars_i, vars_j)
            obj += obj_
            constr += constr_

        if removed_ji[j, i]:
            # print("(j, i) present")
            obj_, constr_ = edge_f((edge[1], edge[0]), vars_j, vars_i)
            # obj += obj_
            constr += constr_
    # print(obj)
    # print(constr)
    return obj, constr


def to_undirected_edge_f(edge_f, adj):
    adj_new, removed_ji, dummy_ij = to_underected_graph(adj)

    new_edge_f = lambda edge, vars_i, vars_j: augmented_edge_f(edge, vars_i, vars_j, edge_f, removed_ji, dummy_ij)

    adj_sym = adj + adj.T
    adj_sym[adj_sym > 0.5] = 1

    return adj_sym + (dummy_ij + dummy_ij.T), new_edge_f
