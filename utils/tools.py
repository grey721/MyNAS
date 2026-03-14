import time
from contextlib import contextmanager

@contextmanager
def timer(label: str = '', verbose: bool = True):
    """
    用法:
        with timer('forward pass'):
            out = net(x)

        with timer() as t:
            ...
        print(t.elapsed)
    """
    class _T:
        elapsed: float = 0.0
        def __str__(self):
            return f'{self.elapsed*1000:.2f} ms'

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
        if verbose:
            tag = f'[{label}] ' if label else ''
            print(f'{tag}{t.elapsed:.6f} s') 