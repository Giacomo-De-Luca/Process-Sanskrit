"""Lemma-mode analyses with the members left open.

Members are attached in exactly one place, `LemmaBuilder._part`, from
`LemmaLexicon.members(lemma)` for a part with one lemma, and nothing in the
analysis reads them back.  So the analysis is run once, with a lexicon that
answers `members` with a mark naming the lemma (`RecordingLexicon`), and each
variant's members are put in afterwards.  `AnalysedPart.member_lemma` is the
lemma whose members lemma mode would attach to the part, or None.

`LibraryAnalyses` keeps the analyses in a JSON-lines store, so a word is
analysed once across runs.  The store's first line is a signature (library
version, analysis algorithm, database file); a store with another signature
is discarded.  A lemma-mode change that alters the analyses bumps neither
`ANALYSIS_ALGORITHM_VERSION` nor anything else here, so it must bump the
library version, or old analyses hide it; while editing under one version,
delete the store.  Members are not stored, so a change to members alone
needs neither.  Words whose
analysis failed are analysed again on the next run.  The library runs in
worker processes with its persistent analysis cache switched off: the
evaluation writes nothing outside its own store.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon

MARK = "\u0000"


@dataclass(frozen=True)
class AnalysedPart:
    form: str
    lemmas: Tuple[str, ...]
    member_lemma: Optional[str]


class RecordingLexicon(LemmaLexicon):
    """A lemma-mode lexicon that records where members go instead of filling them in."""

    def members(self, lemma: str) -> List[str]:
        return [MARK + lemma]


class _Worker:
    """One analyser per process; rebuilt every `RENEW` words to bound its memo."""

    RENEW = 2000
    analyzer = None
    session = None
    count = 0

    @staticmethod
    def initialise_process() -> None:
        """Run in each worker process: no persistent analysis cache there."""
        os.environ["PROCESS_SANSKRIT_CACHE_ENABLED"] = "false"

    @classmethod
    def _analyzer(cls):
        from process_sanskrit.functions.lemmaAnalysis import LemmaAnalyzer, LemmaBuilder
        from process_sanskrit.utils.databaseSetup import get_session

        if cls.session is None:
            cls.session = get_session()
        if cls.analyzer is None or cls.count >= cls.RENEW:
            analyzer = LemmaAnalyzer(session=cls.session, cached=False)
            analyzer.lexicon = RecordingLexicon(cls.session)
            analyzer.builder = LemmaBuilder(analyzer.lexicon)
            cls.analyzer, cls.count = analyzer, 0
        cls.count += 1
        return cls.analyzer

    @classmethod
    def analyse(cls, words: List[str]) -> List[Tuple[str, Optional[list], Optional[str]]]:
        results = []
        for word in words:
            try:
                ## the text is IAST; scheme detection on one short token can misread it
                items = cls._analyzer().analyze(word, input_scheme="IAST")
            except Exception as error:  # a word the library cannot analyse is recorded, not fatal
                results.append((word, None, repr(error)))
                continue
            parts = []
            for item in items:
                for part in item["parts"]:
                    marks = [member[len(MARK):] for member in part.get("members", ()) if member.startswith(MARK)]
                    parts.append([part["form"], list(part["lemmas"]), marks[0] if marks else None])
            results.append((word, parts, None))
        return results


def _analyse_chunk(words: List[str]):
    return _Worker.analyse(words)


class LibraryAnalyses:
    """Lemma-mode analyses of single words, stored as JSON lines."""

    CHUNK = 200
    FORMAT = 1

    def __init__(self, store: Path, workers: int = 0):
        self.store = Path(store)
        self.workers = workers
        self._parts: Dict[str, Tuple[AnalysedPart, ...]] = {}
        self.errors: Dict[str, str] = {}
        self.discarded = False
        self._load()

    @classmethod
    def signature(cls) -> dict:
        """What the stored analyses depend on."""
        from importlib.metadata import PackageNotFoundError, version

        from process_sanskrit.utils.analysisCache import ANALYSIS_ALGORITHM_VERSION
        from process_sanskrit.utils.resourcePaths import get_database_path

        try:
            library = version("process-sanskrit")
        except PackageNotFoundError:
            library = "unknown"
        database = get_database_path().stat()
        return {"store": "compound-member-analyses", "format": cls.FORMAT, "library": library,
                "analysis_algorithm": ANALYSIS_ALGORITHM_VERSION,
                "database": {"bytes": database.st_size, "mtime_ns": database.st_mtime_ns}}

    def _load(self) -> None:
        if not self.store.exists():
            return
        with self.store.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines or self._decode(lines[0]) != self.signature():
            self.discarded = True
            self.store.unlink()
            return
        for line in lines[1:]:
            row = self._decode(line)
            if isinstance(row, list) and len(row) == 3 and row[1] is not None:
                self._remember(*row)

    @staticmethod
    def _decode(line: str):
        """A stored line, or None for a line an interrupted run left unfinished."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _remember(self, word: str, parts: Optional[list], error: Optional[str]) -> None:
        if parts is None:
            self.errors[word] = error or "unknown error"
            parts = []
        else:
            self.errors.pop(word, None)
        self._parts[word] = tuple(AnalysedPart(form, tuple(lemmas), member) for form, lemmas, member in parts)

    def analyse(self, words: Iterable[str]) -> int:
        """Analyse the words not yet in the store; returns how many were analysed."""
        missing = sorted({word for word in words if word and word not in self._parts})
        if not missing:
            return 0
        chunks = [missing[start:start + self.CHUNK] for start in range(0, len(missing), self.CHUNK)]
        self.store.parent.mkdir(parents=True, exist_ok=True)
        new = not self.store.exists()
        with self.store.open("a", encoding="utf-8") as handle:
            if new:
                handle.write(json.dumps(self.signature()) + "\n")
            for results in self._run(chunks):
                for word, parts, error in results:
                    handle.write(json.dumps([word, parts, error], ensure_ascii=False) + "\n")
                    self._remember(word, parts, error)
        return len(missing)

    def _run(self, chunks):
        if self.workers <= 0:
            for chunk in chunks:
                yield _analyse_chunk(chunk)
            return
        with ProcessPoolExecutor(max_workers=self.workers, initializer=_Worker.initialise_process) as pool:
            yield from pool.map(_analyse_chunk, chunks)

    def parts(self, word: str) -> Tuple[AnalysedPart, ...]:
        return self._parts[word]

    def lemma_set(self, word: str) -> FrozenSet[str]:
        """Every lemma lemma mode gives `word`, over all its parts; empty if not analysed."""
        return frozenset(lemma for part in self._parts.get(word, ()) for lemma in part.lemmas)

    def __len__(self) -> int:
        return len(self._parts)
