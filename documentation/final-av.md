# A final `-āv` is `-au`

> **Needs extensive testing.** This fix was measured on the Yoga Sūtra, the 235
> dictionary headwords ending in `v` and a handful of sentences only. Before
> relying on it, test it on a larger annotated corpus (the Digital Corpus of
> Sanskrit, or the WSMP gold of
> [sentence-vs-word-evaluation.md](sentence-vs-word-evaluation.md)): every
> token ending in `-āv`, in all four modes, against the gold lemma.

Before a vowel a final `-au` is written `-āv`: `samādhau eva` → `samādhāv eva`,
`aṣṭau aṅgāni` → `aṣṭāv aṅgāni`.

## Before

The single-word cascade read every final `v` as `u` (`sanskritFixedSandhiMap`),
which is right after a consonant (`gurv` → `guru`, `madhv` → `madhu`, `tv` →
`tu`) and made `samādhāv` into `samādhāu`. No table has that form: no inflected
form ends in `v` (0 noun forms, 0 verb forms) or in `-āu`. A split token got no
reading at all. So every `-āv` word missed the forms table and fell through to
whatever else matched: `samādhāv` → `saṃdha`, or the Monier-Williams root
`samādhāv` (`sam-ā-dhāv`, "to run towards").

## Now

`read_final_sandhi` (`utils/lexicalResources.py`) reads a final `-āv` as `-au`
and any other final letter as before. The single-word cascade
(`WordCascade.resolve`) uses it in place of the bare map. `root_any_word` also
retries a word ending in `-āv` as `-au` (`FINAL_AU_SPELLING`), so a token the
splitter hands on (`samādhāv eva`) is read the same way. Lemma mode keeps the
word as typed as a lemma when Monier-Williams or Apte has it (rule 8 of
[lemma-mode.md](lemma-mode.md)), so `samādhāv` gives `samādhi`, `samādhāv`.

The hybrid cache signature is `hybrid-morphology-v13`: the split path's
morphology changed, so v12 rows must not answer for it.

## What changed

Over 1,008 inputs (every Yoga Sūtra line as a sentence, every distinct Yoga
Sūtra word, the 235 headwords ending in `v`, and 12 probes), 44 `roots`
results, 47 detailed results and 29 lemma results changed; none raises.

| input | before | after |
|---|---|---|
| `te samādhāv upasargā…` (YS 3.37) | `samādhāv` | `samādhi` |
| `…samādhayo 'ṣṭāv aṅgāni` (YS 2.29) | `aṣṭa`/`akṣ` + `av` | `aṣṭa`/`aṣṭi`/`aṣṭan` |
| `nidhāv iti`, `upādhāv iti` | `nidhāv`, `upādhāv` | `nidhi`, `upādhi` |
| `nāv atra` | `nāv` | `na`/`ni`/`nu`/`asmad` (`nau`) |
| `samādhāv` alone | `saṃdha` | `samādhi` |
| `dhāv` alone | `dha` | `dha`/`dhi`/`dhu` |
| `abhidhāv` alone | `abhidhā` | `abhi` + `dha`/`dhi`/`dhu` |

**The cost is a bare root typed alone.** About 25 headwords end in `-āv`, mostly
`dhāv` with preverbs. In `roots` mode they were wrong before and are wrong
differently now. In lemma mode the root stays a lemma beside the new readings,
so a query for `dhāv` finds it, with a few more terms (`dha`, `dhi`, `dhu`). A
bare root does not occur in running text, so an indexed book is not affected.

## Not covered

- Only `-āv`. A final `-av` (the `o` of a vocative before a vowel, `prabhav
  iti`) and a final `-v` on the split path (`gurv atra`) are not read.
- The single-word path still reads `-v` after a consonant as `u` only, never `ū`
  (`vadhv` for `vadhū`).

Pinned by `tests/test_final_av.py`.
