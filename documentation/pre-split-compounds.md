# Pre-split compounds and option forwarding

Callers can hand `process` a compound whose boundaries they already know, using
`-` or `+` as the separator:

```python
process("hetu-pada-adhikam", mode="roots")
# ['hetu', ('pada', 'pad'), ('adhikam', 'adhika')]
```

`handle_special_characters` (in `functions/process.py`) splits on `[-+]` and
re-enters `process` once per segment. The two separators are interchangeable and
may be mixed.

## What the separators actually promise

They pin the boundaries the splitter would otherwise have to guess. They do
**not** mean "treat each segment as final":

- Each segment goes through the *whole* pipeline. A segment that fails root
  lookup is still sandhi-split, so `hetu-padamadhikam` yields
  `hetu, pada, adhikam` — the second segment was split further.
- A segment that cannot be analysed at all contributes **nothing**; it is
  dropped rather than passed through verbatim. `hetu-qqqq` yields `['hetu']`.
- Empty segments (a leading, trailing or doubled separator, as in `hetu-`) are
  skipped.

So the guarantee is "nothing is merged across a boundary you gave", not "no
splitting happens".

## Separators inside a sentence

The pre-split branch lives behind `if ' ' not in text`, so it never sees
multi-word input. A separator in a sentence is therefore normalised to
whitespace in `process` before that gate:

```python
process("yoga-citta-vṛtti nirodhaḥ", mode="roots")
# ['yoga', 'citta', 'vṛtti', 'nirodha']  -- same as the whitespace spelling
```

This is load-bearing, not cosmetic. The splitter does not read `-` or `+` as a
boundary; left in place they reach it verbatim and it **merges the very segments
the caller separated** (`yoga-citta-vṛtti nirodhaḥ` came out as `['yoga',
'cittavṛtti', 'nirodha']`). It does honour whitespace, so demoting the separator
to a space preserves the caller's boundary. Input without a separator is not
touched by this step.

## Two bugs this area used to have

### `-` was silently stripped before it could be seen

`preprocess` used the character class `[^\p{L}'_%*-+]`. Inside a character
class `*-+` is a **range** (`*` through `+`), not three literal characters. `+`
therefore survived while `-` was rewritten to a space — which pushed hyphenated
input into the whitespace branch of `process` and left the pre-split branch
unreachable for `-` entirely. The feature worked for `+` and was dead code for
`-`.

The class is now `[^\p{L}'_%*+\-]`, with `-` escaped and last. **Do not "tidy"
this back into a range.** `tests/test_presplit_options.py` asserts both
characters survive `preprocess`.

### The recursion dropped the caller's options

The three recursive `process()` calls in `handle_special_characters` (asterisk
wildcard, `_`/`%` wildcard, and pre-split) passed only `session`/`_memo`. They
did not forward `mode`, `*dict_names`, `max_length` or `debug`, so the recursion
fell back to the signature defaults. The visible effect:

- `process("hetu-pada", mode="roots")` returned **detailed dictionary entries**.
- `process("hetu-pada", "ap90")` returned only the default **`mw`** dictionary.

Options are now collected into a `forwarded` dict at the top of
`handle_special_characters` and splatted into every recursion. Because the
recursion targets the public entry point, anything added to `process`'s
signature in future must be added to `forwarded` too, or it will be silently
dropped on exactly these paths.

## Wildcards honour `mode` and `dict_names` too

`handle_special_characters` also owns the wildcard forms (`deva*`, `dev_`,
`deva%`). Each has two exits — an early return when the pattern matches a
dictionary entry, and a recursion into `process` when it does not. The early
return, which is the common one, used to hand back raw `dict_search` output
without passing through `clean_results`, so it ignored `mode`:

```python
process("deva*", mode="roots")   # ['deva']   (was: detailed entries)
process("dev_",  mode="roots")   # ['dev_']
process("deva%", mode="roots")   # ['deva%']
```

For a pattern query (`_`, `%`) the stem *is* the literal pattern — a pattern has
no root to speak of. That is deliberate, so don't "fix" it.

## Known wart: `mode="roots"` has two different empty returns

`process("", mode="roots")` returns `""` — a *string* — because of an early
return in `process`. A compound whose segments are all empty (`process("-",
mode="roots")`) returns `[]`. A caller iterating the result therefore gets
characters in one case and stems in the other.

This is pinned by a test rather than fixed: `""` is the published 1.0.x
behaviour for empty input, and changing the return type is a breaking API change
that deserves a deliberate decision rather than being smuggled into a bugfix.

## Tests

`tests/test_presplit_options.py` covers separator survival through `preprocess`,
`mode`/`dict_names` forwarding on both separators, equivalence of a pre-split
compound with the concatenation of its processed segments, `-`/`+` equivalence,
and the empty-segment cases.
