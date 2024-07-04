import builtins

import numpy as np

from graphvx_core.solvers.iteration_schemes.iteration_scheme import IterationScheme, epsilon
from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.iteration_solvers.iteration_solver import IterationSolverX, IterationSolverZ


class ADMMVanillaIteration(IterationScheme):

    def __init__(self):
        super().__init__()
        self.x = None
        self.z = None
        self.u = None
        self.z_old = None
        self.u_old = None
        self.A = None
        self.A_tr = None

        self.run_x = None
        self.run_z = None
        self.iteration_solver_x = None
        self.iteration_solver_z = None

    def prepare(
            self, A, d_x: Dispatcher, d_z: Dispatcher, s_x: IterationSolverX, s_z: IterationSolverZ, x_0=None, z_0=None, u_0=None):
        (nz, nx) = A.shape
        self.A = A
        self.A_tr = A.T
        self.x = np.zeros((nx,), dtype=np.float64)
        self.z = np.zeros((nz,), dtype=np.float64)
        self.u = np.zeros((nz,), dtype=np.float64)
        self.z_old = np.zeros((nz,), dtype=np.float64)
        self.u_old = np.zeros((nz,), dtype=np.float64)

        self.iteration_solver_x = s_x
        self.iteration_solver_z = s_z

        self.run_x = d_x
        self.run_z = d_z

        if x_0 is not None:
            np.copyto(self.x, x_0)
        if z_0 is not None:
            np.copyto(self.z, z_0)
        if u_0 is not None:
            np.copyto(self.u, u_0)

        self.iteration_solver_z.set_dual(self.u)

    def iteration(self, k: int, rho: float):

        # x-update
        self.iteration_solver_x.set_primal(self.z)
        self.run_x(self.iteration_solver_x.gen_chunks())
        self.iteration_solver_x.get_new_primal_values(self.x)

        # z-update
        self.iteration_solver_z.set_primal(self.x)
        self.run_z(self.iteration_solver_z.gen_chunks())
        np.copyto(self.z_old, self.z)
        self.iteration_solver_z.get_new_primal_values(self.z)

        # update dual values
        Ax = self.A.dot(self.x)
        np.copyto(self.u_old, self.u)
        self.u += rho * (Ax - self.z)

        self.iteration_solver_z.set_dual(self.u)

    def check_convergence(self, rho, e_abs, e_rel):
        norm = np.linalg.norm
        Ax = self.A @ self.x
        r = Ax - self.z
        s = rho * self.A_tr.dot(self.z - self.z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = np.sqrt(self.A.shape[1]) * e_abs + e_rel * builtins.max(norm(Ax), norm(self.z)) + epsilon()
        e_dual = np.sqrt(self.A.shape[0]) * e_abs + e_rel * norm(self.A_tr.dot(self.u)) + epsilon()
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return stop, res_pri, e_pri, res_dual, e_dual
