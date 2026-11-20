# Transliteration engine assessment: sanscript vs vidyut-lipi

Assessed 2026-11-10. Two questions:

1. Would vidyut-lipi, the Rust transliterator the Sanskrit Voyager frontend
   compiles to WebAssembly, improve on `indic_transliteration` (sanscript),
   which `transliterate()` and the splitter use here?
2. What would our own Rust library cost that takes the best of both?

## Answers

1. **No, not for the backend.** On the conversions the pipeline depends on, the
   two give the same results. Where they differ, sanscript is the safer one for
   analysis. vidyut is about 7× faster per call, but transliteration is about 6%
   of uncached `process()` time, so that speed barely shows in the pipeline.
2. **A new library covering 10 schemes would cost about 4–6 engineer-weeks
   and is not worth it.**
   The bugs that matter can be fixed by patching vidyut-lipi, which takes under
   a day. The only real gain is one engine on both sides, frontend and backend.
   Getting that from a fork of vidyut-lipi takes about two weeks.

## What was compared

- **sanscript:** `indic_transliteration` 2.3.82, as installed.
- **vidyut:** `vidyut` 0.4.0 from PyPI. Its Rust core is vidyut-lipi, the
  library the frontend uses (it vendors vidyut-lipi 0.2.0). Every probe word
  gave the same output from the wheel and from the frontend's committed WASM
  build.
- **Data:**
  - dictionary headwords from `word_list` (247k);
  - inflected forms from `lgtab2` (150k sampled);
  - running text: 224k distinct Mahābhārata tokens (GRETIL);
  - Devanagari text: 327k Mahābhārata tokens (SARIT).

  80k items of each kind were compared; the patch check below used all 3.52M
  distinct words.

## Accuracy

| Conversion | Where the pipeline uses it | Agreement | Better |
|---|---|---|---|
| IAST → SLP1 | splitter input | 99.96% (words), 99.98% (text) | sanscript. vidyut keeps capitals (`Kṛṣṇa` → `KfzRa`) and turns a medial `oṃ` into the Om sign (`tejoṃśaḥ` → `tejAUMSaH`) |
| HK / SLP1 / ITRANS / WX / Devanagari → IAST | typed input, `preprocess` | same words fail in both | equal |
| Devanagari text → IAST | typed or pasted input | 99.86% | vidyut: `ँ` before a consonant becomes `m̐`; sanscript writes `~` |
| IAST → Devanagari, Kannada, Telugu, Gujarati | `/transliterate` output | 99.72% | sanscript: vidyut deletes letters (see *Bugs*) |
| IAST → Tamil | `/transliterate` output | 31% | vidyut marks aspirates with superscripts (`தா³`), so it reads back losslessly. On words both libraries write alike, sanscript reads plain letters as voiced or aspirated (`ஸாத்மார்பண` → `sādhmārbhaṇa`) and gets 17% back right; vidyut gets 100% |
| IAST → Bengali | `/transliterate` output | 80% | Mixed. vidyut writes `য়` and `ৎ` as Bengali spelling does. Reading Bengali back, sanscript takes `ব` as *v* (95% of words right) and vidyut takes it as *b* (67% right) |
| IAST → Kolkata | `/transliterate` output | — | sanscript only. vidyut has no Kolkata; its ISO 15919 is a different scheme |
| standalone `om` → SLP1 | splitter input | — | vidyut keeps `om`; sanscript gives `AUM` |

**Auto-detection** is no better in vidyut. It also reads SLP1 `Darma` as
Harvard-Kyoto. The figures below are the share of words whose detected scheme
decodes back to the right IAST:

| Input scheme | ours | vidyut |
|---|---|---|
| SLP1 | 79.7% | 79.1% |
| ITRANS | 95.3% | 94.9% |
| Velthuis | 92.2% | 91.7% |
| WX | 19.2% | 34.4% |

The cure for short ASCII input is the explicit `input_scheme` (see
[`input-scheme.md`](input-scheme.md)), which the frontend already sends.

## Speed

| Per call | sanscript | vidyut |
|---|---|---|
| IAST → SLP1 | 8.7 µs | 1.2 µs |
| IAST → Devanagari | 10.3 µs | 1.2 µs |
| SLP1 → IAST | 4.1 µs | 0.6 µs |

Profiling 666 Yoga Sūtra words through `process()`, with the cache off and
under cProfile, put transliteration at 0.21 s of 3.3 s (6%). Where the calls
come from:

- **The splitter's `SanskritObject.transcoded` makes 94% of the calls
  (22,046 of 23,576).** It converts every candidate token from SLP1 to IAST.
  This is vendored reference code pinned by `tests/test_splitter_parity.py`, so
  it must keep sanscript whatever the public function uses.
- **The public `transliterate()` costs about 35 µs per word.** Switching it to
  vidyut would save under 1% of `process()`.

## Bugs found in vidyut-lipi 0.2.0

The frontend displays text with vidyut, so these bugs reach readers. The
details, the patch and the frontend work are in the frontend repository, at
`docs/vidyut-lipi-bugs.md`.

- **A vowel after a vowel sign is lost** when romanised text is written in any
  Brahmic script: `gacchatia` → `गच्छत`, `suāhuta` → `साहुत`.
  - Cause: `had_virama` stays set after a vowel sign replaces the virama, so
    the next vowel pops the sign.
  - This is upstream issue ambuda-org/vidyut#253. It was opened on 2026-09-24,
    has no replies, and describes only the `ā` case.
- **A medial `oṃ` becomes the Om sign:** `tejoṃśaḥ` → `तेज्ॐशः`.
- **How often readers see them:** the frontend's display rules were run over
  all 1,576 books. The two bugs damage about 20,500 displayed words
  (10,200 distinct) in 846 books, 0.07% of all converted words; the Ṛgveda
  editions are worst. 999 of the 246,880 dictionary headwords are damaged too.
- **The fix is two small changes to `transliterate.rs`.**
  - Over 3.52M distinct words, they change 2,222 outputs, and only in Brahmic
    targets; every change is a repair.
  - vidyut's own test suite passes (32 unit tests, 47 integration tests,
    11 doctests).

## Options and cost

Effort is in engineer-days, including tests and docs. Agents can write most of
the code, but they do not remove the slow parts: deciding conventions (`ḷ` vs
ळ, Om, *b*/*v* in Bengali, Tamil superscripts) and reviewing diffs over millions
of words.

| Option | What you get | Effort |
|---|---|---|
| **A. Patch the frontend's vendored vidyut-lipi** | Fixes the display bugs above. No change to this repository. | 0.5–1 day |
| **A′. Convert the analysis box in the frontend** | The analysis box uses the frontend's engine instead of calling this backend's `/transliterate`, so both views show the same script. | ~0.5 day |
| **B. One engine for both sides** (a fork of vidyut-lipi, in its own crate) | <ul><li>The project's input rules move from the frontend's `displayWord.ts` into Rust: lowercasing, `ṁ`, ¨ marks, accents, Kolkata.</li><li>Exposed here through the existing PyO3 extension (`rust/python`) and in the frontend as WASM.</li><li>sanscript stays for the vendored splitter.</li></ul> | 7–10 days |
| **C. Our own engine for 10 schemes** | <ul><li>Tables generated from `common_maps`, the same data both libraries use.</li><li>Its own conversion core (vidyut's is about 2,500 lines, and its bug shows how easy it is to get wrong).</li><li>Detection, PyO3 and WASM bindings, a test suite comparing it with both libraries.</li></ul> | 18–29 days (breakdown below). Maintenance afterwards: about 1–2 days a quarter |

Neither B nor C removes `indic_transliteration`, because the vendored splitter
and its parity tests need it.

### Option C in detail

The 10 schemes are the frontend selector's: IAST, Kolkata, Harvard-Kyoto,
SLP1, WX, ITRANS, Devanagari, Bengali, Kannada and Tamil.

| Part | Days |
|---|---|
| Scheme tables for the 10, generated from `common_maps`; Kolkata is IAST with `ē`/`ō` | 2–3 |
| Conversion core:<ul><li>longest-match reading;</li><li>the abugida rules (virama, inherent *a*, vowel signs, hiatus);</li><li>anusvara, candrabindu, visarga, avagraha and Om;</li><li>Unicode normalization and capitals;</li><li>the project's input rules (`ṁ`, ¨ marks, accents)</li></ul> | 6–10 |
| Detection among the 10 | 2–3 |
| PyO3 binding in `rust/python`, and the WASM C-ABI (the frontend's wrapper is the pattern) | 2–3 |
| Tests: comparison with both libraries over the 3.52M-word corpus, plus fixed cases | 3–5 |
| Switching both sides over, docs, notices, release | 3–5 |
| **Total** | **18–29** |

**Where the cost is:**

- **Fewer scripts barely reduce it.** The core and the tests are about 60% of
  the work, and their size does not depend on the number of scripts. Going from
  15 schemes to 10 saves about 3–6 days.
- **Dropping the 50-script target saves more.** It removes the 10–20 days that
  Thai, Tibetan and Khmer would need for reordering.

**Input outside the 10:** the backend's `detect()` can also return Velthuis,
Telugu, Gujarati, Gurmukhi, Malayalam and Oriya. Input in those schemes would
need to fall back to sanscript, which stays installed anyway.

What option C adds over B is independence from vidyut-lipi's single maintainer.
That part of the library has not changed since November 2024 and has had no
release since January 2025. Option C does not add:

- **Accuracy:** B gets the same fixes.
- **Speed:** vidyut is already about 1 µs per word.

## Licensing

All three upstreams are MIT:

- `indic_transliteration`;
- vidyut-lipi (declared in its `Cargo.toml`; no licence file is shipped);
- the `common_maps` scheme data both are built from.

MIT lets this project copy, fork or adapt any of them under its own licence,
as long as the copyright and permission notice is kept. So option B needs no
clean-room rewrite.

**This project's licence.** Process-Sanskrit is published under
CC-BY-NC-ND-4.0, and `MANIFEST.in` grafts `rust/` into the sdist on PyPI. An
engine written here, whether option B or C, would therefore be public source
that others may read but not modify or use commercially. A fix contributed
upstream to vidyut-lipi stays MIT and reaches everyone who uses it.

**Notices.** `tools/generate_rust_third_party_notices.py` skips local
workspace crates. A vidyut-lipi fork inside the workspace therefore needs its
MIT notice added to `THIRD_PARTY_NOTICES.md` by hand, or the generator
extended to cover it.

## Recommendation

Do **A** now; the patch is written and checked. Offering the patch upstream,
as a fix for vidyut#253, would let the frontend drop its local copy once a
release includes it. Do **A′** if the analysis box
and the book text should match. Choose **B** only if one engine for both sides
becomes a goal of its own. **C** costs 18–29 days, against under a day for A
and under two weeks for B, for no measurable gain.
