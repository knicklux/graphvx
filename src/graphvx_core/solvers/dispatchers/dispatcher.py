from abc import ABC, abstractmethod
from functools import partial

from graphvx_core.solvers.utils import UpdateType


class Dispatcher(ABC):

    @abstractmethod
    def __init__(self, func, update_type: UpdateType, *args, **kwargs):
        self.func = func
        self.update_type = update_type

    @abstractmethod
    def __call__(self, items):
        pass

    @classmethod
    def factory(cls, *args, **kwargs):
        return partial(cls, *args, **kwargs)
