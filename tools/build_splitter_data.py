"""Regenerate the data files used by ``process_sanskrit.splitter``.

The splitter is a vendored, split-only subset of kmadathil/sanskrit_parser
(MIT).  Upstream ships 85 MB of data across seven files, most of which exists
to serve ``Parser.parse()`` and morphological tagging -- neither of which
Process-Sanskrit uses.  This script distils the parts that ``split()`` actually
needs into five legacy files (~23 MB):

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

  log-table.npy      the canonical 1,000-entry float32 log-sigmoid table. It is
                     pinned because transcendental ufunc results differ in
                     their low-order bits across platforms.

  sandhi_rules.zip   copied verbatim from upstream.
  sentencepiece.model  copied verbatim from upstream.

By default the script exports deterministic, language-neutral inputs from the
packaged legacy assets without changing them. The Rust resource builder
consumes those inputs; see ``rust/resource-builder/README.md``.

To deliberately regenerate the legacy assets from an installed
sanskrit_parser==0.2.6, use the separate upstream mode:

    uv pip install sanskrit-parser==0.2.6 gensim sentencepiece marisa-trie
    uv run python tools/build_splitter_data.py --upstream

Regenerating is only necessary if upstream's data changes; the outputs are
committed.  tests/test_splitter_parity.py checks the outputs still reproduce
upstream's behaviour exactly.
"""

import argparse
import json
import pickle
import shutil
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import marisa_trie
import numpy as np

from process_sanskrit.splitter.scorer_model import log_sigmoid_table

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "rust" / "resources.toml"

# Tag ids from sanskrit_util.schema.Tag
NOMINAL, PARTICIPLE = 2, 4
def upstream_data_dir() -> Path:
    import sanskrit_parser

    return Path(sanskrit_parser.__file__).parent / "data"


@dataclass(frozen=True)
class ResourceGenerationConfig:
    """Filesystem contract shared with the Rust resource builder."""

    legacy_data: Path
    neutral_output: Path

    @classmethod
    def from_toml(cls, path: Path):
        try:
            from tomllib import loads
        except ModuleNotFoundError:  # Python 3.9/3.10 regeneration only
            try:
                from toml import loads
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Python 3.9/3.10 resource regeneration requires the "
                    "development-only 'toml' package"
                ) from error

        document = loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("unsupported splitter resource config schema")
        generation = document.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("splitter resource config has no [generation] table")

        def resolve(name):
            configured = Path(generation[name])
            return configured if configured.is_absolute() else path.parent / configured

        return cls(
            legacy_data=resolve("legacy_data"),
            neutral_output=resolve("neutral_output"),
        )


class NativeInputExporter:
    """Write deterministic neutral inputs for the Rust resource builder."""

    SCHEMA_VERSION = 1

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_forms(self, forms) -> None:
        with (self.output_dir / "forms.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            # marisa-trie predictive iteration is deterministic but not byte
            # lexicographic, which the native FST format requires.
            for form in sorted(forms):
                stream.write(form)
                stream.write("\n")

    def write_rules(self, archive: Path) -> tuple[int, int]:
        with ZipFile(archive) as rules_zip:
            with rules_zip.open("sandhi_backward.pkl") as stream:
                backward = pickle.load(stream)

        key_count = 0
        variant_count = 0
        with (self.output_dir / "sandhi-rules.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for after in sorted(backward):
                variants = sorted(
                    {(before[0], before[1]) for before, _annotation in backward[after]}
                )
                record = {
                    "after": after,
                    "variants": [
                        {"left": left, "right": right} for left, right in variants
                    ],
                }
                stream.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                stream.write("\n")
                key_count += 1
                variant_count += len(variants)
        return key_count, variant_count

    def write_scorer(
        self,
        *,
        syn0,
        syn1,
        vocab,
        window,
        cbow_mean,
        codes,
        points,
    ) -> None:
        code_lengths = np.asarray([len(code) for code in codes], dtype=np.uint64)
        point_lengths = np.asarray([len(point) for point in points], dtype=np.uint64)
        code_offsets = np.concatenate(
            (np.zeros(1, dtype=np.uint64), np.cumsum(code_lengths, dtype=np.uint64))
        )
        point_offsets = np.concatenate(
            (np.zeros(1, dtype=np.uint64), np.cumsum(point_lengths, dtype=np.uint64))
        )
        arrays = {
            "syn0": np.asarray(syn0, dtype=np.float32),
            "syn1": np.asarray(syn1, dtype=np.float32),
            "code-offsets": code_offsets,
            "code-data": np.concatenate(codes).astype(np.uint8, copy=False),
            "point-offsets": point_offsets,
            "point-data": np.concatenate(points).astype(np.uint32, copy=False),
            "log-table": log_sigmoid_table(),
        }
        for name, array in arrays.items():
            np.save(self.output_dir / f"{name}.npy", array, allow_pickle=False)

        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "window": int(window),
            "cbow_mean": bool(cbow_mean),
            "vocab": vocab,
        }
        with (self.output_dir / "scorer.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(metadata, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")

    def write_legacy_assets(self, data_dir: Path) -> tuple[int, int, int]:
        """Export the committed Python assets without rewriting them."""
        trie = marisa_trie.Trie()
        trie.load(str(data_dir / "forms.trie"))
        self.write_forms(trie.iterkeys())

        with np.load(data_dir / "w2v.npz", allow_pickle=False) as model:
            codes = np.split(
                model["code_flat"], np.cumsum(model["code_len"], dtype=np.int64)[:-1]
            )
            points = np.split(
                model["point_flat"],
                np.cumsum(model["point_len"], dtype=np.int64)[:-1],
            )
            self.write_scorer(
                syn0=model["syn0"],
                syn1=model["syn1"],
                vocab=model["vocab"].tolist(),
                window=model["window"],
                cbow_mean=model["cbow_mean"],
                codes=codes,
                points=points,
            )
        keys, variants = self.write_rules(data_dir / "sandhi_rules.zip")
        return len(trie), keys, variants


def build_forms_trie(
    src: Path, output_dir: Path, native_exporter: NativeInputExporter
) -> None:
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
    trie.save(str(output_dir / "forms.trie"))
    native_exporter.write_forms(trie.iterkeys())
    size = (output_dir / "forms.trie").stat().st_size / 1e6
    print(f"  forms.trie          {len(forms):>9,} forms "
          f"({generated:,} generated)  {size:.2f} MB")


def build_w2v(src: Path, output_dir: Path, native_exporter: NativeInputExporter) -> None:
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
        output_dir / "w2v.npz",
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
    native_exporter.write_scorer(
        syn0=wv.vectors,
        syn1=model.syn1,
        vocab=vocab,
        window=model.window,
        cbow_mean=model.cbow_mean,
        codes=codes,
        points=points,
    )
    size = (output_dir / "w2v.npz").stat().st_size / 1e6
    print(f"  w2v.npz             {len(vocab):>9,} vocab, dim {model.vector_size}  {size:.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="resource TOML config"
    )
    parser.add_argument(
        "--upstream",
        action="store_true",
        help="regenerate legacy assets from sanskrit_parser before exporting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResourceGenerationConfig.from_toml(args.config.resolve())
    native_exporter = NativeInputExporter(config.neutral_output)

    if not args.upstream:
        forms, keys, variants = native_exporter.write_legacy_assets(
            config.legacy_data
        )
        print(
            f"exported {forms:,} forms, {keys:,} rule keys, and "
            f"{variants:,} unique variants from {config.legacy_data} "
            f"to {config.neutral_output}"
        )
        return

    src = upstream_data_dir()
    config.legacy_data.mkdir(parents=True, exist_ok=True)
    print(f"reading upstream data from {src}")
    build_forms_trie(src, config.legacy_data, native_exporter)
    build_w2v(src, config.legacy_data, native_exporter)
    for name in ("sandhi_rules.zip", "sentencepiece.model"):
        shutil.copy(src / name, config.legacy_data / name)
        print(
            f"  {name:<20}{'':>9} copied verbatim  "
            f"{(config.legacy_data / name).stat().st_size / 1e6:.2f} MB"
        )
    keys, variants = native_exporter.write_rules(src / "sandhi_rules.zip")
    print(
        f"  native inputs       {keys:>9,} rule keys, "
        f"{variants:,} unique variants  {config.neutral_output}"
    )

    total = sum(
        file.stat().st_size for file in config.legacy_data.iterdir() if file.is_file()
    ) / 1e6
    upstream = sum(file.stat().st_size for file in src.iterdir() if file.is_file()) / 1e6
    print(f"\ntotal {total:.1f} MB  (upstream ships {upstream:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
