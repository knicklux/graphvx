from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional


class UpdateType(Enum):
    X = 1
    Z = 2

    def __repr__(self):
        return f"<UpdateType.{self.name}: {self.value}>"

    def __str__(self):
        return self.name


def default_rho_update_func(rho, res_p, thr_p, res_d, thr_d, k):
    return rho


@dataclass
class GraphVXRhoContext:
    __rho_k__: Any = field(default=None, init=False)
    rho_init: float = 1.0 # Optional parameter to initialize rho
    rho: Optional[float] = None  # Optional parameter
    update_rho_enabled: bool = False
    rho_update_func: Callable = field(default=default_rho_update_func, init=False, repr=False)

    def set_rho_container(self, rho_container):
        self.__rho_k__ = rho_container
        if self.rho_init is not None:
            self.__rho_k__.value = self.rho_init

    def get_rho_k(self):
        return float(self.__rho_k__.value)

    def set_rho_k(self, val: Any):
        if self.update_rho_enabled:
            self.__rho_k__.value = val

    def set_rho_update_func(self, func=None):
        self.update_rho_enabled = func is not None
        self.rho_update_func = func if func else default_rho_update_func

    def update_rho(self, res_p, thr_p, res_d, thr_d, k):
        rho_new = self.rho_update_func(self.get_rho_k(), res_p, thr_p, res_d, thr_d, k)
        self.set_rho_k(rho_new)
        return rho_new


rho_context = GraphVXRhoContext()
