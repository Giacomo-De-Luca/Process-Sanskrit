# Analysis pipeline audit

This pass followed the missing `pratipaj` / `balam` investigation. It checked
the active Python pipeline's boundaries: split results, morphology, dictionary
lookup, output cleanup, pre-split input, option forwarding, and cached grammar.
The vendored splitter and native ranking implementation were left unchanged.
The optional BYT5 model was inspected at its adapter boundary but not executed.

## Confirmed fixes

| Failure | Reproduction | Correction |
| --- | --- | --- |
| A completely unresolved token vanished | `process("ḍḍḍḍ", mode="roots")` returned `[]` | `Inflector` treats the compound matcher's empty list as a miss and preserves the token. |
| Missing definitions corrupted morphology | A five-field analysis whose lemma was absent from the dictionary became a nested list in the headword slot; `parts` raised `TypeError` | Preserve the original five fields and append the standard missing-definition payload. |
| Empty dictionary mappings counted as hits | An unmatched `%` / `_` query returned a dictionary name with no definitions | Check the contained mappings, return a stub, and finish wildcard handling without recursively retrying the same pattern. |
| Uppercase dictionary names widened the search | `dict_search(["deva"], "AP90")` unnecessarily added other dictionaries | Match reference-table dictionary names without case sensitivity, preserving caller spelling in output keys. |
| Cleanup lost the request context | Rejoining `anu` + `bhū` after an AP90 lookup fetched MW definitions | Forward the chosen dictionaries and active session through cleanup and its replacement/rejoin lookups. |
| Pre-split `parts` lost its values | `process("hetu-pada", mode="parts")` returned a list of keys | Merge the component mappings, including an empty mapping for separator-only input. |
| `api` produced duplicate and malformed entries | `root_any_word("api")` returned a flat three-item stub; `process("api")` emitted three dictionary entries | Return one five-field particle analysis. The existing `āpi` alias still resolves to `api`, with its input surface retained. |

The two morphology fallback blocks are now one `Inflector` implementation in
`functions/inflect.py`. It shares the request memo and handles prefix-plus-word
attempts before using the same direct/compound fallback for each remaining word.
The `inflect()` function retains its existing callable signature.

The cache signature is `hybrid-morphology-v6`. It invalidates stored grammar
that dropped unresolved tokens or contained malformed `api` entries. Dictionary
payloads remain outside the cache and are rendered with the current request
context. See [local-cache.md](local-cache.md) and
[dictionary-results.md](dictionary-results.md).

## Validation

Regression cases were written and run before implementation. The final suite:

```bash
uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'
```

Result: **300 tests run, 4 skipped, no failures**. The skipped tests cover
optional upstream comparison dependencies and opt-in full native parity. Tests
used a temporary analysis-cache path. The new regressions are in
`tests/test_analysis_result_contracts.py`; `tests/test_analysis_cache.py` also
checks that `api` and an unresolved token survive a real cache round trip.
The earlier `tests/test_compound_inflection.py` tests continue to pass.

A separate before/after run processed **894 unique inputs**: the two bundled
compound benchmark JSON files and the Yoga Sūtra lines, using `cached=False`.
Both runs produced no exceptions, empty result lists, or malformed top-level
entries. Headword sequences changed on **26 inputs**, entirely through removal
of the duplicate `api` entries; other headword sequences were unchanged. This
comparison checks structural integrity and visible changes, and does not
certify the linguistic correctness of every existing analysis.

The required code-quality-reviewer found no blocking regressions and passed
53 focused contract, compound, and cache tests.

## Remaining issues that need separate work

- **`jj` preprocessing and candidate ranking.** The global `jj` → `j j`
  shortcut breaks lexical geminates and can lose the feminine ending in
  `sarvatragāminīpratipaj`. Removing it makes `tajjñānam` regress from
  `tad` + `jñāna` to `tajjña` + `ana`. Correcting this requires ranking-aware
  segmentation rather than deleting the shortcut alone.
- **Position-dependent `-n` replacement.** `clean_results()` skips its final
  entry. Extending that loop applies an existing overbroad replacement rule to
  more words: the corpus then replaces the `ātman` analysis of `ātmanā` with
  the surface headword `ātmanā`, and changes seven other non-`api` cases. The
  loop boundary was retained. The replacement rule needs to preserve attested
  morphology before its iteration range is widened.
- **Unmatched spans and one-character output.** The compound scanner can skip
  unmatched characters inside a partially recognised word. Separately,
  `process()` filters one-character lemmas on its split path: `inflect(["ḍ"])`
  retains the token, while `process("ḍ", mode="roots")` still returns `[]`.
  Preserving completely unresolved tokens in `Inflector` does not change either
  of these older segmentation/filtering policies.
- **Empty roots output type.** `process("", mode="roots")` returns `""`, while
  separator-only roots input returns `[]`. This published convention remains
  pinned in `tests/test_presplit_options.py`; changing it needs an explicit API
  compatibility decision.

These findings are separate from the seven data-flow fixes above and remain
open. None requires a database download or dictionary-row patch.
