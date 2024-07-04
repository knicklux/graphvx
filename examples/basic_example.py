from copy import copy
import warnings
from datetime import datetime

import cvxpy as cp
import networkx as nx
import numpy as np

import graphvx_core as gvx


def update_rho_heuristic(rho, r, thr_p, s,  thr_d, k, tau_inc=1.3, tau_dec=1.3, mu=4, rho_min=1e-6, rho_max=1e6):
    """
    Update the penalty parameter rho based on the primal and dual residuals.

    Parameters:
    - rho (float): Current value of rho.
    - r (float): Primal residual.
    - s (float): Dual residual.
    - k (int): Current iteration number.
    - tau_inc (float): Factor by which to increase rho.
    - tau_dec (float): Factor by which to decrease rho.
    - mu (float): Residual balancing parameter.
    - rho_min (float): Minimum value of rho.
    - rho_max (float): Maximum value of rho.

    Returns:
    - new_rho (float): Updated value of rho.
    """
    if r > mu * s:
        rho = min(tau_inc * rho, rho_max)
    elif s > mu * r:
        rho = max(rho / tau_dec, rho_min)
    return rho


def main():
    digraph = nx.DiGraph(nx.circulant_graph(32, [1]))
    # digraph = nx.DiGraph(nx.circulant_graph(8, [2]))
    # digraph = nx.DiGraph(nx.star_graph(64-1))
    # digraph = nx.DiGraph(nx.barbell_graph(8, 8))
    # digraph = nx.DiGraph(nx.wheel_graph(32-1))

    adj = copy(nx.to_numpy_array(digraph))
    # print(adj)
    # adj[0, 2] = 0
    # digraph = nx.DiGraph(adj)

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
    g.finalize()

    # logging
    logfile = "/tmp/log_graph_sdp_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + ".csv"
    log_residuals_admm = gvx.hooks.log_residuals_to_csv(logfile)
    admm_hooks = {"logging": log_residuals_admm}

    rho_init = 1.0

    def test_rho_update(rho, res_p, thr_p, res_d, thr_d, k):
        return rho_init / np.sqrt(k)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1) create iteration solver
        settings = {}
        i_s_x = gvx.CvxpyADMMX(settings, cp.CLARABEL)
        i_s_z = gvx.CvxpyADMMZ(settings, cp.CLARABEL)
        i_s_x.prepare(g)
        i_s_z.prepare(g)
        # 2) create dispatcher
        n_proc = 8
        chunk_size = 8096
        dispatcher_x = gvx.MultiProcessingDispatcher(i_s_x.solve_handle(), gvx.UpdateType.X, n_proc, chunk_size)
        dispatcher_z = gvx.MultiProcessingDispatcher(i_s_z.solve_handle(), gvx.UpdateType.Z, n_proc, chunk_size)
        # 3) create iteration scheme
        i_s = gvx.ADMMVanillaIteration()

        optimizer = gvx.GraphvxSolver(g, dispatcher_x, dispatcher_z, i_s_x, i_s_z, i_s)

        optimizer.set_rho_update_func(func=None)
        # optimizer.set_rho_update_func(func=test_rho_update)
        # optimizer.set_rho_update_func(func=update_rho_heuristic)

        optimizer.solve(
            rho=rho_init,
            max_iters=500,
            eps_abs=1e-4,
            eps_rel=1e-4,
            verbose=True,
            hooks=admm_hooks
        )

    for i, var in enumerate(variables):
        print(i)
        print(var.value)
        print(var.name())
    print(len(variables))
    print(g.value)


if __name__ == "__main__":
    main()
