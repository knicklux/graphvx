import cvxpy as cp
import numpy as np

from graphvx_core.graphvx import GraphVX


class CvxpyMonolithicSolver:

    def __init__(self, graph: GraphVX, m_func=None):
        super().__init__()

        self.graphvx = graph

        self.status = None
        self.value = 0

        if m_func is None:
            self.m_func = cp.Minimize

    # Adds objectives together to form one collective CVXPY Problem.
    # Option of specifying Maximize() or the default Minimize().
    # Graph status and value properties will also be set.
    # Individual variable values can be retrieved using GetNodeValue().
    # Option to use serial version or distributed ADMM.
    # max_iters optional parameter: Maximum iterations for distributed ADMM.
    def solve(self, m=cp.Minimize, verbose=False, solver=cp.SCS, settings=None):

        global context

        m_func = m

        objective = 0
        constraints = []

        # Add all node objectives and constraints
        # for ni in self.Nodes():
        # nid = ni.GetId()
        # objective += self.node_objectives[nid]
        # constraints += self.node_constraints[nid]
        for (obj_i, constrs_i) in zip(self.graphvx.node_objectives.values(), self.graphvx.node_constraints.values()):
            objective += obj_i
            constraints += constrs_i

        # Add all edge objectives and constraints
        for (obj_ij, constrs_ij) in zip(self.graphvx.edge_objectives.values(), self.graphvx.edge_constraints.values()):
            objective += obj_ij
            constraints += constrs_ij

        # Solve CVXPY Problem
        objective = self.m_func(objective)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=solver, verbose=verbose, **settings)
        except cp.SolverError:
            problem.solve(solver=solver, verbose=verbose, **settings)
        if problem.status in [cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE]:
            problem.solve(solver=solver, verbose=verbose, **settings)

        # Set TGraphVX status and value to match CVXPY
        self.status = problem.status
        self.value = problem.value

        # Insert into hash to support ADMM structures and GetNodeValue()
        for id in self.graphvx.node_ids:

            values = []

            for (varID, varName, var, offset) in self.graphvx.node_variables[id]:

                temp = var.size[0] if hasattr(var.size, '__len__') else var.size
                if temp == 1:
                    val = np.array([var.value])
                else:
                    val = np.array(var.value).reshape(-1, )

                values.append(val)

            self.graphvx.node_values[hash] = values
        self.value = objective.value
        self.graphvx.value = objective.value