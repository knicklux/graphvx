from abc import ABC, abstractmethod

import numpy as np

from graphvx_core.graphvx import GraphVX
from graphvx_core.solvers.utils import UpdateType


class IterationSolverX(ABC):

    def __init__(self, update_type: UpdateType, *args, **kwargs):
        super().__init__()
        self.type = type

    def solve_handle(self):
        return self.solve

    @abstractmethod
    def prepare(self, g: GraphVX):
        pass

    @staticmethod
    @abstractmethod
    def solve(items):
        pass

    @abstractmethod
    def gen_chunks(self):
        pass

    @abstractmethod
    def set_primal(self, p: np.array):
        pass

    @abstractmethod
    def get_new_primal_values(self, p: np.array):
        pass


class IterationSolverZ(IterationSolverX):

    @abstractmethod
    def set_dual(self, d: np.array):
        pass

    @abstractmethod
    def get_new_dual_values(self, d: np.array):
        pass
