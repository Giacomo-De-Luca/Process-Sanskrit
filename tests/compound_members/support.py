"""What the compound-member evaluation tests share."""

from functools import lru_cache
from pathlib import Path

from process_sanskrit.utils.resourcePaths import get_database_path
from scripts.compound_members.dcs import ConlluReader, HeadwordEvidence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmarks" / "compound-members.json"
## Two sentences of Yogasūtrabhāṣya 1.36 as the DCS dump writes them, one with an
## abstract suffix written as a word of its own and a Vedic ḷ written as U+FFB1
## (the dump's stand-in), and one with a participle head and a visarga member.
SAMPLE = Path(__file__).with_name("sample.conllu")


def has_database() -> bool:
    return get_database_path().exists()


@lru_cache(maxsize=1)
def shared_resources():
    """One `Resources` for every test of the package: the bulk tables are read once."""
    from scripts.compound_members.evaluation import Resources

    return Resources()


def sample_sentences():
    with SAMPLE.open(encoding="utf-8") as handle:
        return list(ConlluReader.parse(handle, "Yogasūtrabhāṣya"))


def evidence_of(*splits, whole=0, surfaces=None, exact=None) -> HeadwordEvidence:
    """Evidence for one headword: `(lemmas, forms)` splits, whole occurrences, tokens."""
    evidence = HeadwordEvidence()
    evidence.whole = whole
    for lemmas, forms in splits:
        evidence.splits[(tuple(lemmas), tuple(forms))] += 1
    evidence.surfaces.update(surfaces or {})
    evidence.exact_surfaces.update(exact or {})
    return evidence


class FakeAnalyses:
    """Stands in for `LibraryAnalyses` with hand-written parts."""

    def __init__(self, parts):
        self._parts = parts

    def parts(self, word):
        return self._parts[word]
