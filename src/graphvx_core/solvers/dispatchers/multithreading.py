from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, wait
from itertools import batched

from graphvx_core.solvers.dispatchers.dispatcher import Dispatcher
from graphvx_core.solvers.utils import UpdateType


class MultiThreadingDispatcher(Dispatcher):
    def __init__(self, func, update_type, num_processors, chunk_size=8096):
        super().__init__(func, update_type)
        self.num_processors = num_processors
        self.chunk_size = chunk_size

    def __call__(self, items):
        print(f"Threaded {self.update_type} update in progress for {len(items)} items")
        with ThreadPoolExecutor(max_workers=self.chunk_size) as executor:
            for chunk in batched(items, self.chunk_size):
                futures = {}
                for index, lentry in enumerate(chunk):
                    future = executor.submit(self.func, lentry)
                    futures[future] = index
                wait(list(futures.keys()))
            executor.shutdown()
