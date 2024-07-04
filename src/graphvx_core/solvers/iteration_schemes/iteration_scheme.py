from abc import ABC, abstractmethod

import numpy as np

from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.iteration_solvers.iteration_solver import IterationSolverX, IterationSolverZ

epsilon_ = np.sqrt(np.finfo(np.float32).eps)


class IterationScheme(ABC):

    def __init__(self):
        super().__init__()
        self.x = None
        self.z = None
        self.u = None

    @abstractmethod
    def prepare(
            self, A, d_x: Dispatcher, d_z: Dispatcher, s_x: IterationSolverX, s_z: IterationSolverZ, x_0=None, z_0=None, u_0=None):
        pass

    @abstractmethod
    def iteration(self, k: int, rho: float):
        pass

    @abstractmethod
    def check_convergence(self, rho, e_abs, e_rel):
        pass


def epsilon():
    global epsilon_
    return epsilon_
