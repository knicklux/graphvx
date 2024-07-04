from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched

from tqdm import tqdm

from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.utils import UpdateType


class MultiProcessingDispatcher(Dispatcher):
    def __init__(self, func, update_type, num_processors, chunk_size=8096):
        super().__init__(func, update_type)
        self.num_processors = num_processors
        self.chunk_size = chunk_size

    def __call__(self, items):
        print(f"Distributed {self.update_type} update in progress for {len(items)} items")
        with ProcessPoolExecutor(max_workers=self.num_processors) as executor:
            with tqdm(total=len(items)) as progress_bar:
                for chunk in batched(items, self.chunk_size):
                    futures = {}
                    for index, lentry in enumerate(chunk):
                        future = executor.submit(self.func, lentry)
                        futures[future] = index
                    for _ in as_completed(futures):
                        progress_bar.update(1)
            executor.shutdown()
