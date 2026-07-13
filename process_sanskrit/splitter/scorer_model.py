"""Parity-critical constants shared by scorer runtime and asset export."""

import numpy as np


# gensim/models/word2vec_inner.pyx
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
# Integer division matches Cython's DEF constant folding. This must not be 83.33.
INDEX_SCALE = EXP_TABLE_SIZE // MAX_EXP // 2


def log_sigmoid_table() -> np.ndarray:
    """Return gensim's float32 hierarchical-softmax lookup table."""
    indices = np.arange(EXP_TABLE_SIZE, dtype=np.float32)
    exponent = np.exp(
        (indices / np.float32(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP
    ).astype(np.float32)
    return np.log((exponent / (exponent + 1)).astype(np.float32)).astype(
        np.float32
    )
