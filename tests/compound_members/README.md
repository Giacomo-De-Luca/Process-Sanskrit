# Tests of the compound-member evaluation

These tests cover `scripts/compound_members/`, the evaluation that compares ways
of reading a compound headword's members against the Digital Corpus of
Sanskrit. They pin the evaluation's own code: the DCS reader, the scorers and
the variants. The member reading lemma mode ships has its own tests, in
`tests/test_compound_members.py`, because it is library code.

Run them all from the repository root:

```bash
uv run python -m unittest discover -s tests/compound_members -t .
```

Tests that need the lexicon database skip without it. None needs the DCS dump:
`sample.conllu` stands in for it.

## Files

| file | what it pins |
|---|---|
| `support.py` | shared helpers: the committed configuration's path, the CoNLL-U sample, `evidence_of` (hand-built `HeadwordEvidence`), `FakeAnalyses` |
| `sample.conllu` | three DCS sentences, written as the dump writes them: a multiword token, a suffix word (`tvāt`), the U+FFB1 stand-in for `ḷ`, a participle head, a visarga member |
| `test_dcs.py` | `ConlluReader` (tokens, words, suffix folding, parsed twins) and `DcsEvidence` (whole and split occurrences, sandhi between segments, participle heads, `max_run`) |
| `test_lexicon.py` | `Spelling` keys, `JoinSpellings`, and that the bulk reads (`ComponentTables`, `Headwords`) give what the library's lookups give |
| `test_readings.py` | each option of the brief on hand-checked headwords; `baseline` is the rule before 2026-11-22, and `conservative` is exactly lemma mode's `LemmaLexicon.members` |
| `test_scoring.py` | `MatchRule` (the search's rule), `MemberScorer` (both yardsticks), `CaseBook`, `RetrievalSimulation` (reach and the weighted comparison), `RunningTextImpact` |
| `test_analyses.py` | that recorded analyses with the library's members reproduce `process(mode="lemmas")`, and the analysis store (signature, retries, an unfinished line) |
| `test_configuration.py` | the committed configuration and case book, and that every configured variant builds |
