"""DCS word2vec scorer for sandhi splits, without gensim.

Upstream scores candidate splits with a CBOW / hierarchical-softmax word2vec
model trained on the Digital Corpus of Sanskrit, loaded through gensim.  The
scoring itself is a forward pass -- no training, no optimiser -- so gensim is
carried purely to unpickle the weights and run ~20 lines of arithmetic.  That
costs a compiled dependency (and scipy, and smart_open), and gensim currently
publishes no wheel for Python 3.14, which pins the whole project's ceiling.

tools/build_splitter_data.py exports the weights to a plain .npz, and the pass
is reimplemented below in numpy.  This is a faithful port, not an improvement:
it reproduces gensim's output to within float32 rounding (max |delta| ~1e-4 over
the Yoga Sutra; see tests/test_splitter_parity.py), *including* two quirks that
a from-scratch implementation would not have:

  - gensim skips any term with |f| >= MAX_EXP instead of computing it, so very
    confident predictions contribute exactly 0 rather than ~0.
  - the sigmoid is a 1000-entry lookup table, and the index scale is
    ``EXP_TABLE_SIZE / MAX_EXP / 2`` evaluated with *integer* division: 83, not
    83.33.  Using the true quotient shifts every score by ~0.07/token.

Reproducing them matters because the scores rank splits, and Process-Sanskrit
re-ranks on top (functions/sandhiSplitScorer.py).  Drifting here would silently
change which split wins.  See gensim/models/word2vec_inner.pyx,
score_pair_cbow_hs.
"""

import logging

import numpy as np

from .data_manager import data_file_path

logger = logging.getLogger(__name__)

# gensim/models/word2vec_inner.pyx
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
# NOTE: integer division, matching Cython's DEF constant folding. Not 83.33.
_INDEX_SCALE = EXP_TABLE_SIZE // MAX_EXP // 2


def _log_sigmoid_table() -> np.ndarray:
    i = np.arange(EXP_TABLE_SIZE, dtype=np.float32)
    e = np.exp((i / np.float32(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP).astype(np.float32)
    return np.log((e / (e + 1)).astype(np.float32)).astype(np.float32)


class Scorer:
    """Log-probability of a split under the DCS word2vec model.

    Unlike upstream, this does not fall back to a length heuristic when the model
    is unavailable -- see _load().
    """

    def __init__(self):
        self._model = None
        self._sp = None
        self._enabled = None

    def _load(self) -> bool:
        """Load the model, or raise.

        Upstream degrades gracefully here: if gensim/sentencepiece are missing it
        logs a warning, sets gensim_enabled = False, and silently ranks splits by
        length instead of likelihood. That was reasonable when scoring was an
        optional extra -- but it is the single worst failure mode this package
        has, because the splitter keeps working and just gets quietly worse, and
        process_sanskrit/__init__.py silences this logger outright, so the warning
        would never reach anyone.

        Scoring is not optional here: sentencepiece and numpy are hard
        dependencies. A scorer that cannot load is a broken install, so say so.
        """
        if self._enabled:
            return True
        try:
            import sentencepiece as spm

            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(data_file_path("sentencepiece.model"))

            z = np.load(data_file_path("w2v.npz"))
            self._syn0 = z["syn0"]
            self._syn1 = z["syn1"]
            # vocab is stored in row order, so position == row in syn0/syn1.
            self._index = {w: i for i, w in enumerate(z["vocab"])}
            self._window = int(z["window"])
            self._cbow_mean = int(z["cbow_mean"])
            self._code = np.split(z["code_flat"], np.cumsum(z["code_len"])[:-1])
            self._point = np.split(z["point_flat"], np.cumsum(z["point_len"])[:-1])
            self._log_table = _log_sigmoid_table()
            self._enabled = True
        except Exception as e:
            raise RuntimeError(
                "The sandhi split scorer failed to load, so splits cannot be "
                "ranked by DCS likelihood. Splitting would still run, but would "
                "silently produce worse splits, so this is fatal instead. "
                "Check that sentencepiece is installed and that the splitter's "
                "data files (sentencepiece.model, w2v.npz) shipped with the "
                f"package. Underlying error: {e}"
            ) from e
        return True

    def _score_pieces(self, pieces) -> float:
        ids = [self._index[p] for p in pieces if p in self._index]
        n = len(ids)
        total = np.float32(0.0)
        dim = self._syn0.shape[1]

        for pos, word in enumerate(ids):
            lo = max(0, pos - self._window)
            hi = min(n, pos + self._window + 1)
            context = [ids[m] for m in range(lo, hi) if m != pos]

            neu1 = np.zeros(dim, dtype=np.float32)
            if context:
                neu1 = self._syn0[context].sum(axis=0, dtype=np.float32)
                if self._cbow_mean:
                    neu1 = (neu1 * np.float32(1.0 / len(context))).astype(np.float32)

            f = (neu1 @ self._syn1[self._point[word]].T).astype(np.float32)
            f = f * ((-1.0) ** self._code[word]).astype(np.float32)

            keep = (f > -MAX_EXP) & (f < MAX_EXP)  # gensim skips saturated terms
            idx = ((f[keep] + MAX_EXP) * _INDEX_SCALE).astype(np.int32)
            total += self._log_table[idx].sum(dtype=np.float32)

        return float(total)

    def score_splits(self, splits) -> list:
        return self.score_strings([" ".join(map(str, s)) for s in splits])

    def score_strings(self, sentences) -> list:
        self._load()
        return [self._score_pieces(self._sp.EncodeAsPieces(s)) for s in sentences]
