"""Parity-critical constants shared by scorer runtime and asset export."""

import numpy as np

from .data_manager import data_file_path


# gensim/models/word2vec_inner.pyx
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
# Integer division matches Cython's DEF constant folding. This must not be 83.33.
INDEX_SCALE = EXP_TABLE_SIZE // MAX_EXP // 2


def log_sigmoid_table() -> np.ndarray:
    """Return the pinned gensim-compatible hierarchical-softmax table.

    NumPy's transcendental ufuncs can differ by a few float32 low-order bits
    across operating systems. Loading the committed table keeps Python scoring
    and native resource regeneration byte-identical on every build host.
    """
    table = np.load(data_file_path("log-table.npy"), allow_pickle=False)
    if table.shape != (EXP_TABLE_SIZE,) or table.dtype != np.dtype(np.float32):
        raise RuntimeError(
            "The pinned splitter log-sigmoid table must contain exactly "
            f"{EXP_TABLE_SIZE} float32 entries"
        )
    if not np.isfinite(table).all():
        raise RuntimeError("The pinned splitter log-sigmoid table is not finite")
    return table
