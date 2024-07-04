from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import numbers
from copy import copy
import multiprocessing

import cvxpy as cp
import numpy as np

from graphvx_core.graphvx import GraphVX
from graphvx_core.hooks import *
from graphvx_core.vectorizer.cvxpy_vectorizer import *
from graphvx_core.vectorizer.numpy_vectorizer import *
from graphvx_core.solvers.utils import UpdateType, rho_context
from graphvx_core.solvers.iteration_solvers.iteration_solver import IterationSolverX, IterationSolverZ
from graphvx_core.graphvx import X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS, X_VECTIZER
from graphvx_core.graphvx import Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND
from graphvx_core.graphvx import Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND


@dataclass
class GraphVXSolverContextX:

    settings: dict = field(default_factory=dict)
    solver: Any = cp.CLARABEL

    m_func: Callable = field(default=cp.Minimize, init=False)

    node_vals: Any = None


@dataclass
class GraphVXSolverContextZ:

    settings: dict = field(default_factory=dict)
    solver: Any = cp.CLARABEL

    m_func: Callable = field(default=cp.Minimize, init=False)

    edge_z_vals: Any = None
    edge_u_vals: Any = None


admm_context_x = GraphVXSolverContextX()
admm_context_z = GraphVXSolverContextZ()


class CvxpyADMMX(IterationSolverX):

    def __init__(self, settings=None, solver=None, m_func=cp.Minimize, *args, **kwargs):

        super().__init__(UpdateType.X, *args, **kwargs)

        admm_context_x.m_func = m_func
        admm_context_x.settings = settings or admm_context_x.settings
        admm_context_x.solver = solver or admm_context_x.solver
        self.node_lentries = []
        self.edge_list = []
        self.x_length = 0
        self.z_length = 0

        self.manager = multiprocessing.Manager()

    def prepare(self, g: GraphVX):

        if rho_context.update_rho_enabled:
            rho_used = None
        else:
            rho_used = rho_context.rho_init

        for node in g.node_list:
            # packing
            problem, rho_p, z_params, u_params, undo = ADMM_x_problem(node, rho=rho_used)
            problem_data = (problem, rho_p, z_params, u_params, None)
            lentry = (node, problem_data)
            self.node_lentries.append(lentry)

        self.edge_list = g.edge_list
        self.x_length = g.x_length
        self.z_length = g.z_length

        admm_context_x.node_vals = multiprocessing.Array('d', np.zeros((self.x_length,), dtype=np.float64))

    @staticmethod
    def solve(items):
        ADMM_x_lambda(items)

    def gen_chunks(self):
        return self.node_lentries

    def set_primal(self, p: np.array):

        for edge in self.edge_list:

            for (varID, varName, var, offset) in edge[Z_IVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = p[edge[Z_ZIJIND] + offset: edge[Z_ZIJIND] + offset + size]
                setValue(admm_context_z.edge_z_vals, edge[Z_ZIJIND] + offset, size, value)

            for (varID, varName, var, offset) in edge[Z_JVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = p[edge[Z_ZJIIND] + offset: edge[Z_ZJIIND] + offset + size]
                setValue(admm_context_z.edge_z_vals, edge[Z_ZJIIND] + offset, size, value)

    def get_new_primal_values(self, p: np.array):
        # target = np.zeros((self.x_length,), dtype=np.float64)
        for (node, _) in self.node_lentries:
            for (varID, varName, var, offset) in node[X_VARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = getValue(admm_context_x.node_vals, node[X_IND], size)
                p[node[X_IND] + offset: node[X_IND] + offset + size] = value


class CvxpyADMMZ(IterationSolverZ):

    def __init__(self, settings=None, solver=None, m_func=cp.Minimize, *args, **kwargs):
        super().__init__(UpdateType.Z, *args, **kwargs)
        admm_context_z.settings = settings or admm_context_z.settings
        admm_context_z.solver = solver or admm_context_z.solver
        admm_context_z.m_func = m_func
        self.edge_lentries = []
        self.node_list = []
        self.x_length = 0
        self.z_length = 0

        self.manager = multiprocessing.Manager()

    def prepare(self, g: GraphVX):

        if rho_context.update_rho_enabled:
            rho_used = None
        else:
            rho_used = rho_context.rho_init

        for edge in g.edge_list:
            # packing
            problem, rho_p, params_x_i, params_x_j, params_u_ij, params_u_ji, undo_i, undo_j = ADMM_z_problem(
                edge, rho=rho_used)
            problem_data = (problem, rho_p, params_x_i, params_x_j, params_u_ij, params_u_ji, None, None)
            lentry = (edge, problem_data)
            self.edge_lentries.append(lentry)

        self.node_list = g.node_list
        self.x_length = g.x_length
        self.z_length = g.z_length

        admm_context_z.edge_z_vals = multiprocessing.Array('d', np.zeros((self.z_length,), dtype=np.float64))
        admm_context_z.edge_u_vals = multiprocessing.Array('d', np.zeros((self.z_length,), dtype=np.float64))

    @staticmethod
    def solve(items):
        ADMM_z_lambda(items)

    def gen_chunks(self):
        return self.edge_lentries

    def set_primal(self, p: np.array):
        for node in self.node_list:
            for (varID, varName, var, offset) in node[X_VARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = p[node[X_IND] + offset: node[X_IND] + offset + size]
                setValue(admm_context_x.node_vals, node[X_IND] + offset, size, value)

    def get_new_primal_values(self, p: np.array):
        # target = np.zeros((self.z_length,), dtype=np.float64)

        for (edge, _) in self.edge_lentries:

            for (varID, varName, var, offset) in edge[Z_IVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = getValue(admm_context_z.edge_z_vals, edge[Z_ZIJIND] + offset, size)
                p[edge[Z_ZIJIND] + offset: edge[Z_ZIJIND] + offset + size] = value

            for (varID, varName, var, offset) in edge[Z_JVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = getValue(admm_context_z.edge_z_vals, edge[Z_ZJIIND] + offset, size)
                p[edge[Z_ZJIIND] + offset: edge[Z_ZJIIND] + offset + size] = value

    def get_new_dual_values(self, d: np.array):
        # target = np.zeros((self.z_length,), dtype=np.float64)

        for (edge, _) in self.edge_lentries:

            for (varID, varName, var, offset) in edge[Z_IVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = getValue(admm_context_z.edge_u_vals, edge[Z_UIJIND] + offset, size)
                d[edge[Z_UIJIND] + offset: edge[Z_UIJIND] + offset + size] = value

            for (varID, varName, var, offset) in edge[Z_JVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = getValue(admm_context_z.edge_u_vals, edge[Z_UJIIND] + offset, size)
                d[edge[Z_UJIIND] + offset: edge[Z_UJIIND] + offset + size] = value

    def set_dual(self, d: np.array):

        for (edge, _) in self.edge_lentries:

            for (varID, varName, var, offset) in edge[Z_IVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = d[edge[Z_UIJIND] + offset: edge[Z_UIJIND] + offset + size]
                setValue(admm_context_z.edge_u_vals, edge[Z_UIJIND] + offset, size, value)

            for (varID, varName, var, offset) in edge[Z_JVARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                value = d[edge[Z_UJIIND] + offset: edge[Z_UJIIND] + offset + size]
                setValue(admm_context_z.edge_u_vals, edge[Z_UJIIND] + offset, size, value)


def ADMM_x_problem(entry, rho=None, m_func=cp.Minimize):
    variables = entry[X_VARS]
    norms = 0
    inner_term = 0

    if rho is None:
        rho_p = cp.Parameter((), "rho_{}".format(entry[X_NID]), nonneg=True)
    else:
        rho_p = rho

    z_params = []
    u_params = []
    undo_i = []

    for i in range(entry[X_DEG]):

        # define parameters
        z_params_i = []
        u_params_i = []
        for j, (varID, varName, var, offset) in enumerate(variables):
            temp = var.size[0] if hasattr(var.size, '__len__') else var.size
            z_params_i.append(cp.Parameter((temp,), "admm_x_pz_{}_{}_{}".format(entry[X_NID], i, j)))
            u_params_i.append(cp.Parameter((temp,), "admm_x_pu_{}_{}_{}".format(entry[X_NID], i, j)))
        z_params.append(z_params_i)
        u_params.append(u_params_i)

        # Add norm for Variables corresponding to the node
        undo = []
        for j, (varID, varName, var, offset) in enumerate(variables):
            z = z_params[i][j]
            u = u_params[i][j]
            # vectorize var to compute norm
            var_vec, undo_ = cpVectorizer.auto(var)
            undo.append(undo_)
            norms += cp.sum_squares(var_vec - z)
            inner_term += u @ var_vec.T
            # norms += cp.sum_squares(var_vec - z + u)
            # norms += cp.square(cp.norm(var_vec - z + u))
        undo_i.append(undo)

    objective = copy(entry[X_OBJ]) + inner_term + (rho_p / 2) * norms
    objective = m_func(objective)
    constraints = entry[X_CON]
    problem = cp.Problem(objective, constraints)

    return problem, rho_p, z_params, u_params, undo_i


# Helper function to solve the x-update for ADMM for each node
def ADMM_x(lentry, solver, settings):
    global rho_context
    global admm_context_x
    global admm_context_z
    rho = rho_context.get_rho_k()

    # unpacking
    entry, problem_data = lentry
    problem, rho_p, z_params, u_params, undo = problem_data

    variables = entry[X_VARS]

    if not isinstance(rho_p, numbers.Number):
        rho_p.value = rho

    # Iterate through all neighbors of the node to fill values
    for i in range(entry[X_DEG]):
        z_index = X_NEIGHBORS + (2 * i)
        u_index = z_index + 1
        zi = entry[z_index]
        ui = entry[u_index]
        # Add norm for Variables corresponding to the node
        for j, (varID, varName, var, offset) in enumerate(variables):
            temp = var.size[0] if hasattr(var.size, '__len__') else var.size
            z_ = getValue(admm_context_z.edge_z_vals, zi + offset, temp)
            u_ = getValue(admm_context_z.edge_u_vals, ui + offset, temp)
            z = z_params[i][j]
            u = u_params[i][j]
            z.value = z_
            u.value = u_

    try:
        problem.solve(solver, **settings)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)
    if problem.status in [cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE]:
        print("ECOS error: using SCS for x update")
        problem.solve(solver=cp.SCS)

    # Write back result of x-update
    writeObjective(admm_context_x.node_vals, entry[X_IND], problem.objective, variables, undo)

    return 0


def ADMM_z_problem(entry, rho=None, m_func=cp.Minimize):
    norms = 0
    inner_term = 0

    objective = entry[Z_OBJ]
    constraints = entry[Z_CON]

    if rho is None:
        rho_p = cp.Parameter((), "rho_{}".format(entry[Z_EID]), nonneg=True)
    else:
        rho_p = rho

    params_x_i = []
    params_x_j = []
    params_u_ij = []
    params_u_ji = []
    undo_i = []
    undo_j = []

    variables_i = entry[Z_IVARS]
    for i, (varID, varName, var, offset) in enumerate(variables_i):
        temp = var.size[0] if hasattr(var.size, '__len__') else var.size
        x_i = cp.Parameter((temp,), "admm_z_px_i_{}".format(entry[X_NID], i))
        params_x_i.append(x_i)
        u_ij = cp.Parameter((temp,), "admm_z_pu_ij_{}".format(entry[X_NID], i))
        params_u_ij.append(u_ij)
        # vectorize variable
        var_vec, undo = cpVectorizer.auto(var)
        undo_i.append(undo)
        norms += cp.sum_squares(x_i - var_vec)
        inner_term += -var_vec @ u_ij
        # norms += cp.sum_squares(x_i - var_vec + u_ij)
        # norms += cp.square(cp.norm(x_i - var_vec + u_ij))

    variables_j = entry[Z_JVARS]
    for i, (varID, varName, var, offset) in enumerate(variables_j):
        temp = var.size[0] if hasattr(var.size, '__len__') else var.size
        x_j = cp.Parameter((temp,), "admm_z_px_j_{}".format(entry[X_NID], i))
        params_x_j.append(x_j)
        u_ji = cp.Parameter((temp,), "admm_z_pu_ji_{}".format(entry[X_NID], i))
        params_u_ji.append(u_ji)
        # vectorize variable
        var_vec, undo = cpVectorizer.auto(var)
        undo_j.append(undo)
        norms += cp.sum_squares(x_j - var_vec)
        inner_term += -var_vec @ u_ji
        # norms += cp.sum_squares(x_j - var_vec + u_ji)
        # norms += cp.square(cp.norm(x_j - var_vec + u_ji))

    objective = m_func(objective + inner_term + (rho_p / 2) * norms)
    problem = cp.Problem(objective, constraints)

    return problem, rho_p, params_x_i, params_x_j, params_u_ij, params_u_ji, undo_i, undo_j


# Helper function to solve the z-update for ADMM for each edge
def ADMM_z(lentry, solver, settings):
    global rho_context
    global admm_context_x
    global admm_context_z
    rho = rho_context.get_rho_k()

    # unpacking
    entry, problem_data = lentry
    problem, rho_p, params_x_i, params_x_j, params_u_ij, params_u_ji, undo_i, undo_j = problem_data

    variables_i = entry[Z_IVARS]
    for i, (varID, varName, var, offset) in enumerate(variables_i):
        temp = var.size[0] if hasattr(var.size, '__len__') else var.size
        x_i = getValue(admm_context_x.node_vals, entry[Z_XIIND] + offset, temp)
        u_ij = getValue(admm_context_z.edge_u_vals, entry[Z_UIJIND] + offset, temp)
        params_x_i[i].value = x_i
        params_u_ij[i].value = u_ij

    variables_j = entry[Z_JVARS]
    for i, (varID, varName, var, offset) in enumerate(variables_j):
        temp = var.size[0] if hasattr(var.size, '__len__') else var.size
        x_j = getValue(admm_context_x.node_vals, entry[Z_XJIND] + offset, temp)
        u_ji = getValue(admm_context_z.edge_u_vals, entry[Z_UJIIND] + offset, temp)
        params_x_j[i].value = x_j
        params_u_ji[i].value = u_ji

    if not isinstance(rho_p, numbers.Number):
        rho_p.value = rho

    try:
        problem.solve(solver, **settings)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)
    if problem.status in [cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE]:
        print("ECOS error: using SCS for z update")
        problem.solve(solver=cp.SCS)

    # Write back result of z-update. Must write back for i- and j-node
    writeObjective(admm_context_z.edge_z_vals, entry[Z_ZIJIND], problem.objective, variables_i, undo_i)
    writeObjective(admm_context_z.edge_z_vals, entry[Z_ZJIIND], problem.objective, variables_j, undo_j)

    return 0

def ADMM_x_lambda(e):
    global context
    return ADMM_x(e, admm_context_x.solver, admm_context_x.settings)


def ADMM_z_lambda(e):
    global context
    return ADMM_z(e, admm_context_z.solver, admm_context_z.settings)


# Extract a numpy array value from a shared Array.
# Give shared array, starting index, and total length.
def getValue(shared, idx, size):
    return np.array(shared[idx:idx+size])


# Write value of numpy array nparr (with given length) to a shared Array at
# the given starting index.
def setValue(shared, idx, size, val):
    shared[idx:idx+size] = np.asarray(val)


# Write the values for all of the Variables involved in a given Objective to
# the given shared Array.
# variables should be an entry from the node_values structure.
# def writeObjective(shared, idx, objective, variables, devectorize):
#     for v in objective.variables():
#         for (varID, varName, var, offset) in variables:
#             if varID == v.id:
#                 # vectorize variable
#                 setValue(shared, idx + offset, npVectorizer.auto(np.array(v.value))[0])
#                 break
def writeObjective(shared, idx, objective, variables, devectorize):
    for (varID, varName, var, offset) in variables:
        size = var.size[0] if hasattr(var.size, '__len__') else var.size
        setValue(shared, idx + offset, size, npVectorizer.auto(np.array(var.value))[0])
