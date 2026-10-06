import numpy as np

def purged_walk_forward(n, n_splits=5, embargo=30, min_train=0.4):
    """Expanding window. Train stops `embargo` rows before test starts,
    so no training label (which looks `embargo` rows ahead) overlaps test."""
    start = int(n * min_train)
    edges = np.linspace(start, n, n_splits + 1, dtype=int)
    for k in range(n_splits):
        ts, te = edges[k], edges[k + 1]
        yield np.arange(0, max(ts - embargo, 0)), np.arange(ts, te)
