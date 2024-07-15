import multiprocessing
import sys
from collections import defaultdict

import numpy as np
import cvxpy as cp

from graphvx_core.graphvx import GraphVX
from graphvx_core.hooks import *
from graphvx_core.vectorizer.cvxpy_vectorizer import *
from graphvx_core.vectorizer.numpy_vectorizer import *
from graphvx_core.solvers.utils import rho_context
from graphvx_core.solvers.iteration_solvers.iteration_solver import IterationSolverX, IterationSolverZ
from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.iteration_schemes.iteration_scheme import IterationScheme


class GraphvxSolver:
    """
    split into:
        [x] iteration scheme
        [x] node solver
        [x] edge solver
        [x] node dispatcher
        [x] edge dispatcher
    iteration schemes:
        [x] vanilla
        [x] accelerated
        [x] accelerated with reset
        [x] accelerated2
    node and edge solver utilities:
        [x] ADMM
        [x] override (used for custom solver or prox-op, tested with built-in --> implement IterationSolver yourself)
        [ ] monolithic-cvxpy
        [ ] built-in (SDP solver written for this project, uses cvxpy to produce standard form)
        [ ] cuda
        [ ] open-cl (easy, if cuda is based on cupy)
    node and edge dispatcher:
        [x] single
        [x] multiprocessing
        [ ] async (used for protoype gpu accelerated, batched KKT solving)
        [ ] celery (should be similar to multiprocessing)

    later: compatibility-matrix check
    """

    def __init__(self,
            graph: GraphVX,
            dispatcher_x: Dispatcher,
            dispatcher_z: Dispatcher,
            iteration_solver_x: IterationSolverX,
            iteration_solver_z: IterationSolverZ,
            iteration_scheme: IterationScheme
        ):
        super().__init__()

        self.graphvx = graph
        self.dispatcher_x = dispatcher_x
        self.dispatcher_z = dispatcher_z
        self.iteration_solver_x = iteration_solver_x
        self.iteration_solver_z = iteration_solver_z
        self.iteration_scheme = iteration_scheme

    def set_rho_update_func(self, func=None):
        rho_context.set_rho_update_func(func)

    # Implementation of distributed ADMM
    # Uses a global value of rho_param for rho
    # Will run for a maximum of max_iters iterations
    def solve(self,
              rho=1.0,
              max_iters=250,
              eps_abs=1e-3,
              eps_rel=1e-3,
              verbose=False,
              hooks=None):

        if not hooks:
            hooks = {}

        manager = multiprocessing.Manager()
        rho_context.set_rho_container(manager.Value("c_double", rho_context.rho_init, lock=True))
        rho_context.set_rho_k(rho)
        rho = rho_context.get_rho_k()

        all_hooks = defaultdict(noop_factory(), **hooks)

        # examine structure of the constraints
        A = self.graphvx.A
        A_ar = A.toarray()
        print(f"Non-zero entries in constraints {np.count_nonzero(A_ar > 0)} of {A_ar.shape[0] * A_ar.shape[1]}")
        print(f"Condition number in constraints {np.linalg.cond(A_ar)}")

        self.iteration_scheme.prepare(
            A, self.dispatcher_x, self.dispatcher_z, self.iteration_solver_x, self.iteration_solver_z)

        num_iterations = 0

        for num_iter in range(max_iters):

            # run iteration
            self.iteration_scheme.iteration(num_iter, rho)

            # check for convergence
            print(f"Iteration: {num_iter}, rho: {rho}")

            stop, res_pri, e_pri, res_dual, e_dual = self.iteration_scheme.check_convergence(rho, eps_abs, eps_rel)
            if verbose:
                # Debugging information to print convergence criteria values
                print('  r:', res_pri, A.shape[1])
                print('  e_pri:', e_pri)
                print('  s:', res_dual, A.shape[0])
                print('  e_dual:', e_dual)

            x = self.iteration_scheme.x
            z = self.iteration_scheme.z
            u = self.iteration_scheme.u
            iteration_done_info = {"num_iterations": num_iter, "res_pri": res_pri, "res_dual": res_dual,
                                   "e_pri": e_pri, "e_dual": e_dual, "stop": stop, "x": x, "z": z, "u": u}
            all_hooks["logging"](iteration_done_info)

            if stop:
                num_iterations = num_iter
                break

            # update rho for next iteration
            if rho_context.update_rho_enabled:
                rho = rho_context.update_rho(res_pri, e_pri, res_dual, e_dual, num_iter + 1)

            num_iterations += 1

        self.graphvx.save_solution(num_iterations, max_iters, self.iteration_scheme.x)
