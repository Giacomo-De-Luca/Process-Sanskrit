"""Regenerate the data files used by ``process_sanskrit.splitter``.

The splitter is a vendored, split-only subset of kmadathil/sanskrit_parser
(MIT).  Upstream ships 85 MB of data across seven files, most of which exists
to serve ``Parser.parse()`` and morphological tagging -- neither of which
Process-Sanskrit uses.  This script distils the parts that ``split()`` actually
needs into three files (~23 MB):

  forms.trie         validity oracle.  Upstream answers "is this a real Sanskrit
                     form?" from inria_forms_pos.db (28 MB) + sanskrit_data.db
                     (17 MB) via sqlalchemy, plus inria_stems_tags_buf.pkl
                     (33 MB) which is only read by get_tags().  Splitting only
                     ever calls valid(), so we precompute the accept set.

                     Inria's half is a plain form table.  sanskrit_data's half
                     is partly *generative*: SimpleAnalyzer._analyze_as_stem
                     strips a nominal ending and looks for a matching stem, so
                     it accepts forms that appear in no table (~40% of accepted
                     forms on the Yoga Sutra).  We enumerate that accept set by
                     running the rule forwards over every stem x ending pair.

  w2v.npz            the DCS word2vec scorer, exported from gensim's pickle so
                     it can be loaded with numpy alone.  See splitter/scorer.py.

  sandhi_rules.zip   copied verbatim from upstream.
  sentencepiece.model  copied verbatim from upstream.

Run against an installed sanskrit_parser==0.2.6:

    pip install sanskrit-parser==0.2.6 gensim sentencepiece marisa-trie
    python tools/build_splitter_data.py

Regenerating is only necessary if upstream's data changes; the outputs are
committed.  tests/test_splitter_parity.py checks the outputs still reproduce
upstream's behaviour exactly.
"""

import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import marisa_trie
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "process_sanskrit" / "splitter" / "data"

# Tag ids from sanskrit_util.schema.Tag
NOMINAL, PARTICIPLE = 2, 4
# gensim word2vec_inner.pyx
EXP_TABLE_SIZE, MAX_EXP = 1000, 6


def upstream_data_dir() -> Path:
    import sanskrit_parser

    return Path(sanskrit_parser.__file__).parent / "data"


def build_forms_trie(src: Path) -> None:
    """Enumerate every form CombinedWrapper.valid() accepts, into one trie."""
    from sanskrit_util import sounds

    con = sqlite3.connect(src / "sanskrit_data.db")

    gender_set = defaultdict(set)
    for group_id, gender_id in con.execute(
        "SELECT group_id, gender_id FROM gender_group_assocs"
    ):
        gender_set[group_id].add(gender_id)

    # Mirror SimpleAnalyzer.__init__'s ending table exactly.
    endings = []
    for name, stem_type, gender_id in con.execute(
        "SELECT name, stem_type, gender_id FROM nominal_ending"
    ):
        # Endings are looked up by suffix length starting at 1, so a zero-length
        # ending can never match. Upstream stores 10 of them; they are dead rows.
        if not name:
            continue
        is_consonant_stem = stem_type[-1] in sounds.CONSONANTS
        if stem_type == "_":
            stem_type, is_consonant_stem = "", True
        endings.append((name, stem_type, gender_id, is_consonant_stem))
        if "n" in name:
            # __init__ registers an n -> R retroflex variant of every ending.
            endings.append((name.replace("n", "R"), stem_type, gender_id, is_consonant_stem))

    by_stem_type = defaultdict(list)
    for e in endings:
        by_stem_type[e[1]].append(e)

    forms = set()
    for stem, genders_id, pos_id in con.execute(
        "SELECT name, genders_id, pos_id FROM stem"
    ):
        if not stem:
            continue
        genders = gender_set.get(genders_id, set())
        # _analyze_as_stem rejects a consonant-stem ending whose stem would end
        # in a vowel or be a bare consonant.
        blocked = stem[-1] in sounds.VOWELS or stem in sounds.CONSONANTS
        for stem_type, group in by_stem_type.items():
            if stem_type and not stem.endswith(stem_type):
                continue
            base = stem[: len(stem) - len(stem_type)] if stem_type else stem
            for name, _st, gender_id, is_consonant_stem in group:
                if is_consonant_stem and blocked:
                    continue
                if pos_id == NOMINAL and gender_id not in genders:
                    continue
                # Upstream drops every feminine ending for participles: its
                # guard `e.stem_type != 'at' or e.stem_type != 't'` is a
                # tautology. Faithfully reproduced -- do not "fix".
                if pos_id == PARTICIPLE and gender_id == 2:
                    continue
                forms.add(base + name)
    generated = len(forms)

    forms.update(r[0] for r in con.execute("SELECT name FROM form"))
    inria = sqlite3.connect(src / "inria_forms_pos.db")
    forms.update(r[0] for r in inria.execute("SELECT form FROM forms"))
    forms.discard("")
    forms.discard(None)

    trie = marisa_trie.Trie(forms)
    trie.save(str(OUT / "forms.trie"))
    size = (OUT / "forms.trie").stat().st_size / 1e6
    print(f"  forms.trie          {len(forms):>9,} forms "
          f"({generated:,} generated)  {size:.2f} MB")


def build_w2v(src: Path) -> None:
    """Export gensim's Word2Vec pickle to a plain numpy archive."""
    import gensim

    model = gensim.models.Word2Vec.load(str(src / "word2vec_model.dat"))
    if not model.hs or model.sg:
        raise SystemExit("expected a CBOW hierarchical-softmax model")
    wv = model.wv

    vocab = list(wv.index_to_key)  # position == row in syn0/syn1
    codes = [np.asarray(wv.get_vecattr(w, "code"), dtype=np.uint8) for w in vocab]
    points = [np.asarray(wv.get_vecattr(w, "point"), dtype=np.uint32) for w in vocab]

    np.savez_compressed(
        OUT / "w2v.npz",
        syn0=wv.vectors.astype(np.float32),
        syn1=model.syn1.astype(np.float32),
        vocab=np.array(vocab),
        window=np.int32(model.window),
        cbow_mean=np.int8(model.cbow_mean),
        code_flat=np.concatenate(codes),
        code_len=np.array([len(c) for c in codes], dtype=np.int32),
        point_flat=np.concatenate(points),
        point_len=np.array([len(p) for p in points], dtype=np.int32),
    )
    size = (OUT / "w2v.npz").stat().st_size / 1e6
    print(f"  w2v.npz             {len(vocab):>9,} vocab, dim {model.vector_size}  {size:.2f} MB")


def main() -> None:
    src = upstream_data_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"reading upstream data from {src}")

    build_forms_trie(src)
    build_w2v(src)
    for name in ("sandhi_rules.zip", "sentencepiece.model"):
        shutil.copy(src / name, OUT / name)
        print(f"  {name:<20}{'':>9} copied verbatim  "
              f"{(OUT / name).stat().st_size / 1e6:.2f} MB")

    total = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file()) / 1e6
    upstream = sum(f.stat().st_size for f in src.iterdir() if f.is_file()) / 1e6
    print(f"\ntotal {total:.1f} MB  (upstream ships {upstream:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
