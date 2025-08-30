"""Helpers (logging, timing, etc.)"""

import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start
