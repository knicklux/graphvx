from tqdm import tqdm

from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.utils import UpdateType


class SequentialDispatcher(Dispatcher):
    def __init__(self, func, update_type: UpdateType, *args, **kwargs):
        super().__init__(func, update_type)

    def __call__(self, items):
        print(f"Sequential {self.update_type} update in progress for {len(items)} items")
        with tqdm(total=len(items)) as progress_bar:
            for item in items:
                self.func(item)
                progress_bar.update(1)
