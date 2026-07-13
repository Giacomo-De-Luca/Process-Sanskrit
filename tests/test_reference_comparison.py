"""A/B comparison between the legacy app-root parser and the current library.

The app-root reference (references/app-root/app/processSanskrit.py) cannot be
imported directly (it needs Flask + a PostgreSQL database), so its parsing
strategies are re-implemented here in IAST on top of the library's own
infrastructure.  That lets each of the four behavioural changes be toggled
in isolation while everything else is held constant:

1. Compound cuts: longest-match (app-root) vs scored + 0.6 gate (library).
   Toggled by monkeypatching ``compoundAnalysis.dict_word_iterative``.
2. Compound lexicon: MW-only key list (app-root) vs all-dictionary
   ``DICTIONARY_REFERENCES`` (library).  Toggled by monkeypatching
   ``compoundAnalysis.DICTIONARY_REFERENCES``.
3. Prefix stripping: one-level, only in process() (app-root) vs nested,
   inside root_any_word (library).  Toggled by monkeypatching
   ``rootAnyWord.SANSKRIT_PREFIXES``.
4. Internal script: SLP1 tables (app-root) vs IAST tables (library).
   Checked by extracting the app-root tables with ast and transliterating.

Run the unittest suite:      .venv/bin/python -m unittest tests.test_reference_comparison
Generate the full report:    .venv/bin/python -m tests.test_reference_comparison --report
"""

import ast
import importlib.util
import json
import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

from indic_transliteration.sanscript import transliterate as _sanscript_tr

import process_sanskrit.functions.compoundAnalysis as compound_analysis
import process_sanskrit.functions.rootAnyWord as root_any_word_module
from process_sanskrit.functions.compoundAnalysis import (
    evaluate_compound_split,
    root_compounds,
)
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import get_session
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils import lexicalResources
from process_sanskrit.utils.lexicalResources import SANDHI_VARIATIONS

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT_SOURCE = REPO_ROOT / "references" / "app-root" / "app" / "processSanskrit.py"
MW_KEYS_PATH = REPO_ROOT / "process_sanskrit" / "resources" / "MWKeysOnly.json"
REPORT_PATH = REPO_ROOT / "tests" / "results" / "reference_comparison_report.txt"

# app-root SANSKRIT_PREFIXES keys (processSanskrit.py:990) converted SLP1->IAST.
# Note: contains 'ā' but NOT bare 'a'; the library added 'a', 'saṃ', 'praty', 'vy'.
APP_ROOT_PREFIXES = [
    'sam', 'anu', 'abhi', 'ati', 'adhi', 'apa', 'api', 'ava', 'ā', 'ud',
    'upa', 'nis', 'parā', 'pari', 'pra', 'prati', 'vi', 'ut', 'ni',
]


def _load_dataset(name):
    path = REPO_ROOT / "tests" / "datasets" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


# ---------------------------------------------------------------------------
# Legacy (app-root) strategy ports, in IAST
# ---------------------------------------------------------------------------

def legacy_try_match_with_prefixes(word, lexicon):
    """Port of app-root try_match_with_prefixes (processSanskrit.py:1012)."""
    if word in lexicon:
        return (word, word[-1])
    for prefix in sorted(APP_ROOT_PREFIXES, key=len, reverse=True):
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            if remainder in lexicon:
                return (word, word[-1])
    return None


def make_legacy_matcher(lexicon):
    """Port of app-root dict_word_iterative (processSanskrit.py:1050).

    Pure longest-match: the longest suffix-trimmed string found in the
    lexicon (directly, via prefixes, or via sandhi variants) wins,
    unconditionally.  No scoring, no minimum-score gate.
    """

    def legacy_dict_word_iterative(word, min_score=None, session=None,
                                   debug=False, _memo=None):
        temp_word = word
        best_match = None
        best_length = 0

        root_result = root_any_word(word, session=session, _memo=_memo)
        if root_result:
            return (word, word[-1])

        while temp_word and len(temp_word) > 1:
            if temp_word in lexicon and len(temp_word) > best_length:
                best_match = temp_word
                best_length = len(temp_word)

            prefix_match = legacy_try_match_with_prefixes(temp_word, lexicon)
            if prefix_match and len(prefix_match[0]) > best_length:
                best_match = prefix_match[0]
                best_length = len(prefix_match[0])

            last_char = temp_word[-1]
            if last_char in SANDHI_VARIATIONS:
                for variant in SANDHI_VARIATIONS[last_char]:
                    test_word = temp_word[:-1] + variant
                    if test_word in lexicon and len(test_word) > best_length:
                        best_match = test_word
                        best_length = len(test_word)
                    prefix_match = legacy_try_match_with_prefixes(test_word, lexicon)
                    if prefix_match and len(prefix_match[0]) > best_length:
                        best_match = prefix_match[0]
                        best_length = len(prefix_match[0])

            if best_match and len(best_match) == len(temp_word):
                break
            temp_word = temp_word[:-1]

        if best_match:
            return (best_match, word[len(best_match) - 1])
        return None

    return legacy_dict_word_iterative


def app_root_style_root(word, session=None):
    """app-root single-word rooting: no prefixes inside root_any_word,
    one-level prefix loop applied afterwards (processSanskrit.py:1416-1426)."""
    with patched(root_any_word_module, 'SANSKRIT_PREFIXES', []):
        result = root_any_word(word, session=session, _memo={})
        if result is not None:
            return result
        for prefix in sorted(APP_ROOT_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix):
                remainder = word[len(prefix):]
                attempt = root_any_word(remainder, session=session, _memo={})
                if attempt is not None:
                    return [prefix] + attempt
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stems(result):
    """Extract the ordered, deduplicated stem candidates from a
    root_any_word result (a list of per-match lists and/or bare strings)."""
    if not result:
        return None
    out = []
    for match in result:
        if isinstance(match, list) and match:
            out.append(str(match[0]))
        elif isinstance(match, str):
            out.append(match)
    seen = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def segment_matches_gold(segment, gold_word, session):
    if segment == gold_word or segment.startswith(gold_word):
        return True
    analysis = root_any_word(segment, session=session, _memo={})
    return bool(analysis) and gold_word in (stems(analysis) or [])


def classify_split(segments, gold, session):
    if not segments:
        return 'no_analysis'
    if len(segments) == 1 and len(gold) > 1:
        return 'whole_word'
    if len(segments) == len(gold) and all(
        segment_matches_gold(s, g, session) for s, g in zip(segments, gold)
    ):
        return 'gold'
    return 'other'


def extract_app_root_tables():
    """Pull the SLP1 sandhi tables out of app-root with ast (the module
    itself cannot be imported) and transliterate them to IAST."""
    source = APP_ROOT_SOURCE.read_text()
    tree = ast.parse(source)
    wanted = {
        'variableSandhiSLP1', 'sanskritFixedSandhiMapSLP1',
        'VOWEL_SANDHI_INITIALS', 'SANDHI_VARIATIONS',
    }
    tables = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in wanted):
            # keep the last definition, which is the one in scope at runtime
            tables[node.targets[0].id] = ast.literal_eval(node.value)

    def slp1_to_iast(value):
        if isinstance(value, str):
            return _sanscript_tr(value, 'slp1', 'iast')
        if isinstance(value, list):
            return [slp1_to_iast(item) for item in value]
        if isinstance(value, dict):
            return {slp1_to_iast(k): slp1_to_iast(v) for k, v in value.items()}
        return value

    return {name: slp1_to_iast(table) for name, table in tables.items()}


# ---------------------------------------------------------------------------
# Unittest: hard invariants for each change
# ---------------------------------------------------------------------------

class ReferenceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = get_session()
        with open(MW_KEYS_PATH, encoding='utf-8') as f:
            cls.mw_keys = set(json.load(f))

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    # -- change 1: scored cuts ---------------------------------------------

    def test_scored_gate_rejects_over_split_ka_ending(self):
        """The gate's real job: refuse a cut that only matches because a
        derivational ending was swallowed.  'devaka' is a dictionary hit, but
        stripping the 'ka' leaves the valid word 'deva', so the cut is an
        over-split and is penalised below the threshold.  app-root's
        longest-match had no such check."""
        score = evaluate_compound_split('devaka', 'qqq', session=self.session)
        self.assertLess(score, 0.6)

    def test_no_shorter_cut_can_outscore_a_longer_one_on_a_junk_tail(self):
        """Regression guard for the shorter-cut pathology.

        The balance bonus used to be keyed to the TAIL (>= 2 chars), so on
        'devaq' the shorter cut 'dev' (junk tail 'aq', 0.4 + 0.2) beat the
        longer 'deva' (junk tail 'q', 0.4).  Keying the bonus to the tail's
        plausibility instead had the same disease one level up — it let 'he'
        beat 'hetu' on 'heturdvividhe'.

        The bonus now depends on first_part ALONE, so equally-evidenced cuts
        tie and the longest wins (the cursor walks longest-first).  'devaq'
        resolves to 'deva', the same answer app-root's longest-match gave."""
        for tail in ('q', 'aq', 'vaq'):
            first = 'devaq'[:len('devaq') - len(tail)]
            self.assertAlmostEqual(
                evaluate_compound_split(first, tail, session=self.session),
                0.6,
                msg=f"{first} + {tail} must tie, not outrank its neighbours",
            )

        scored = compound_analysis.dict_word_iterative(
            'devaq', session=self.session, _memo={}
        )
        legacy = make_legacy_matcher(DICTIONARY_REFERENCES)(
            'devaq', session=self.session, _memo={}
        )
        self.assertEqual(scored[0], 'deva')
        self.assertEqual(legacy[0], 'deva')

        # and the longer cut still wins where the tail is merely awkward
        self.assertEqual(
            root_compounds('heturdvividhe', session=self.session),
            ['hetu', 'dvividhe'],
        )

    def test_whole_word_dictionary_hit_is_accepted(self):
        """An exact headword match leaves no remainder to justify, so the
        remainder credit is satisfied.  Without this, dictionary words missing
        from the forms DB — like the bare stem vṛtti — scored below the gate
        and fragmented into junk cuts like vṛ + tti."""
        self.assertAlmostEqual(
            evaluate_compound_split('vṛtti', '', session=self.session), 1.0
        )
        matched = compound_analysis.dict_word_iterative(
            'vṛtti', session=self.session, _memo={}
        )
        self.assertEqual(matched[0], 'vṛtti')
        segments = root_compounds('cittavṛtti', session=self.session)
        self.assertEqual(segments, ['cittavṛtti'])

    def test_vowel_restoration_recovers_sandhi_cut(self):
        """The remainder of a cut is scored with the initial vowels that
        vowel sandhi may have swallowed (rājapuruṣo = rājapuruṣa + u...), so
        the long cut no longer loses to 'rāja' + 'puruṣottamaḥ'."""
        segments = root_compounds('rājapuruṣottamaḥ', session=self.session)
        self.assertEqual(segments, ['rājapuruṣa', 'uttamaḥ'])

    def test_scored_cut_matches_gold_on_flagged_compound(self):
        segments = root_compounds('samādhipariṇāmaḥ', session=self.session)
        self.assertEqual(segments, ['samādhi', 'pariṇāmaḥ'])

    # -- change 2: lexicon ---------------------------------------------------

    def test_lexicon_is_broader_than_mw(self):
        self.assertGreater(len(DICTIONARY_REFERENCES), len(self.mw_keys))

    def test_membership_lookup_is_faster_than_list_scan(self):
        """app-root membership was `word in <194k-element list>` — O(n).
        The library's DICTIONARY_REFERENCES is an indexed lookup."""
        mw_list = sorted(self.mw_keys)  # a list, as app-root loaded it
        probes = ['yogaścittavṛtti', 'kaścit', 'zzz', 'nirodha',
                  'dharmakṣetra', 'qqq'] * 20

        start = time.perf_counter()
        for word in probes:
            word in mw_list  # noqa: B015 — timing the linear scan
        list_time = time.perf_counter() - start

        start = time.perf_counter()
        for word in probes:
            word in DICTIONARY_REFERENCES
        dict_time = time.perf_counter() - start

        self.assertLess(dict_time, list_time / 2)

    # -- change 3: prefix stripping ------------------------------------------

    def test_samadhi_is_not_oversplit_by_prefixes(self):
        """The word the app-root author flagged ('breaks a lot of words like
        samadhi') must still resolve to itself: direct lookup wins before the
        prefix loop can fire."""
        result = root_any_word('samādhi', session=self.session, _memo={})
        self.assertIsNotNone(result)
        # the forms DB resolves it to the stem samādhin; what matters is that
        # no prefix-stripped fragment (sam / ā / dhi ...) appears instead
        for stem in stems(result):
            self.assertTrue(
                stem.startswith('samādh'),
                f"prefix loop produced spurious stem {stem!r}",
            )

    def test_prefixed_word_gains_analysis_over_app_root(self):
        """Words only analyzable via nested prefixes should now resolve."""
        library = root_any_word('samudāgama', session=self.session, _memo={})
        legacy = app_root_style_root('samudāgama', session=self.session)
        # the library should do at least as well as app-root on this word
        if legacy is not None:
            self.assertIsNotNone(library)

    def test_prefix_stripping_does_not_shadow_whole_word_match(self):
        """Prefix stripping inside a sandhi-variant recursion must not
        preempt a whole-word match reachable through a later variant:
        utkrāntiś resolves via the ḥ-variant to the stem utkrānti, not to
        ut + krānti."""
        self.assertEqual(
            stems(root_any_word('utkrāntiś', session=self.session, _memo={})),
            ['utkrānti'],
        )
        self.assertEqual(
            stems(root_any_word(
                'apratisaṃkramāyās', session=self.session, _memo={}
            )),
            ['apratisaṃkrama'],
        )

    def test_privative_prefix_gain_is_preserved(self):
        """Deferring prefixes must not lose the analyses they enable when no
        whole-word match exists anywhere: asaṃsargaḥ -> a + saṃsarga."""
        self.assertEqual(
            stems(root_any_word('asaṃsargaḥ', session=self.session, _memo={})),
            ['a', 'saṃsarga'],
        )

    # -- change 4: sandhi table parity ----------------------------------------

    @unittest.skipUnless(
        APP_ROOT_SOURCE.exists(),
        f"app-root reference not present at {APP_ROOT_SOURCE}",
    )
    def test_sandhi_tables_match_app_root_modulo_documented_additions(self):
        ## references/ is gitignored, so this reference is absent on a fresh
        ## clone and in CI.  Skip rather than error, matching how
        ## test_splitter_parity treats the optional upstream sanskrit-parser.
        tables = extract_app_root_tables()

        lib_variable = dict(lexicalResources.variableSandhi)
        # library additions on top of app-root:
        for added_key in ('c', 'ā'):
            lib_variable.pop(added_key, None)
        self.assertEqual(tables['variableSandhiSLP1'], lib_variable)

        lib_fixed = dict(lexicalResources.sanskritFixedSandhiMap)
        # 'ś'->'ḥ' was a hardcoded special case in app-root process()
        # (processSanskrit.py:1405), folded into the map by the library.
        lib_fixed.pop('ś', None)
        self.assertEqual(tables['sanskritFixedSandhiMapSLP1'], lib_fixed)

        self.assertEqual(
            tables['VOWEL_SANDHI_INITIALS'], lexicalResources.VOWEL_SANDHI_INITIALS
        )

        lib_variations = {k: list(v) for k, v in SANDHI_VARIATIONS.items()}
        # library adds 'u' as a variation of final 'o'
        self.assertIn('u', lib_variations['o'])
        lib_variations['o'] = [v for v in lib_variations['o'] if v != 'u']
        self.assertEqual(tables['SANDHI_VARIATIONS'], lib_variations)


# ---------------------------------------------------------------------------
# Report mode: full corpus diffs for manual evaluation
# ---------------------------------------------------------------------------

def run_compound_configurations(words, session, mw_keys):
    """Run root_compounds over `words` under the three configurations."""
    results = {}
    for text in words:
        results[text] = {}
        # (1) library: scored cuts, full lexicon
        results[text]['scored_full'] = root_compounds(
            text, session=session, _memo={}
        )
        # (2) app-root strategy: longest match, full lexicon (isolates change 1)
        with patched(compound_analysis, 'dict_word_iterative',
                     make_legacy_matcher(DICTIONARY_REFERENCES)):
            results[text]['longest_full'] = root_compounds(
                text, session=session, _memo={}
            )
        # (3) scored cuts, MW-only lexicon (isolates change 2)
        with patched(compound_analysis, 'DICTIONARY_REFERENCES', mw_keys):
            results[text]['scored_mw'] = root_compounds(
                text, session=session, _memo={}
            )
    return results


def report(out=sys.stdout):
    session = get_session()
    with open(MW_KEYS_PATH, encoding='utf-8') as f:
        mw_keys = set(json.load(f))

    test_cases = _load_dataset('testCases.py').test_cases
    ys = _load_dataset('yogaSutra.py').ys
    with open(REPO_ROOT / 'tests' / 'datasets' /
              'sanskrit_compounds_benchmark.json', encoding='utf-8') as f:
        benchmark = json.load(f)['compounds']

    w = out.write

    # ----- changes 1 + 2: gold-labelled compound splits ----------------------
    w("=" * 100 + "\n")
    w("CHANGES 1 & 2 — COMPOUND CUTS, gold-labelled cases (tests/datasets/testCases.py)\n")
    w("configs: scored_full = library | longest_full = app-root strategy on same lexicon"
      " | scored_mw = library strategy on MW-only lexicon\n")
    w("=" * 100 + "\n")

    gold_words = [case['input'] for case in test_cases]
    gold_results = run_compound_configurations(gold_words, session, mw_keys)

    tallies = {name: {} for name in ('scored_full', 'longest_full', 'scored_mw')}
    for case in test_cases:
        text, gold = case['input'], case['correct_split']
        row = gold_results[text]
        classes = {}
        for name, segments in row.items():
            cls = classify_split(segments, gold, session)
            classes[name] = cls
            tallies[name][cls] = tallies[name].get(cls, 0) + 1
        interesting = len({tuple(v or []) for v in row.values()}) > 1
        marker = '  <-- configs disagree' if interesting else ''
        w(f"\n{text}   (gold: {' + '.join(gold)}, {case['type']}/{case['complexity']}){marker}\n")
        for name in ('scored_full', 'longest_full', 'scored_mw'):
            w(f"    {name:13s} [{classes[name]:11s}] {row[name]}\n")

    w("\nSUMMARY over %d gold cases (gold = matches labelled split; whole_word = kept"
      " as one word; other = different split):\n" % len(test_cases))
    for name, counts in tallies.items():
        w(f"    {name:13s} {dict(sorted(counts.items()))}\n")

    # ----- changes 1 + 2: unlabelled benchmark sample -------------------------
    w("\n" + "=" * 100 + "\n")
    w("CHANGES 1 & 2 — BENCHMARK COMPOUNDS (no gold labels, showing only disagreements)\n")
    w("=" * 100 + "\n")
    sample = [c['text'] for c in benchmark['short'][:40]] + \
             [c['text'] for c in benchmark['medium'][:20]]
    bench_results = run_compound_configurations(sample, session, mw_keys)
    disagreements = 0
    for text in sample:
        row = bench_results[text]
        if len({tuple(v or []) for v in row.values()}) > 1:
            disagreements += 1
            w(f"\n{text}\n")
            for name in ('scored_full', 'longest_full', 'scored_mw'):
                w(f"    {name:13s} {row[name]}\n")
    w(f"\n{disagreements}/{len(sample)} sampled compounds get different splits"
      " across the three configurations.\n")

    # ----- change 3: prefix stripping over Yoga Sutra vocabulary --------------
    w("\n" + "=" * 100 + "\n")
    w("CHANGE 3 — PREFIX STRIPPING, root_any_word over unique Yoga Sutra tokens\n")
    w("library = nested prefixes inside root_any_word | app_root = one-level loop"
      " outside root_any_word\n")
    w("=" * 100 + "\n")

    tokens = set()
    for line in ys:
        for token in line.split():
            token = token.strip()
            if token.startswith("'"):
                token = 'a' + token[1:]
            if token:
                tokens.add(token)

    same = both_none = 0
    library_only, legacy_only, different = [], [], []
    for token in sorted(tokens):
        lib_stems = stems(root_any_word(token, session=session, _memo={}))
        legacy_stems = stems(app_root_style_root(token, session=session))
        if lib_stems == legacy_stems:
            if lib_stems is None:
                both_none += 1
            else:
                same += 1
        elif lib_stems and not legacy_stems:
            library_only.append((token, lib_stems))
        elif legacy_stems and not lib_stems:
            legacy_only.append((token, legacy_stems))
        else:
            different.append((token, lib_stems, legacy_stems))

    w(f"\n{len(tokens)} unique tokens: {same} same analysis, {both_none} unanalyzed"
      f" by both, {len(library_only)} analyzed only by library,"
      f" {len(legacy_only)} only by app-root, {len(different)} different.\n")

    if library_only:
        w("\nAnalyzed ONLY by the library (nested prefixes) — check each for"
          " spurious prefix-stripping:\n")
        for token, lib_stems in library_only:
            w(f"    {token:35s} -> {lib_stems}\n")
    if legacy_only:
        w("\nAnalyzed ONLY by app-root style (should be rare):\n")
        for token, legacy_stems in legacy_only:
            w(f"    {token:35s} -> {legacy_stems}\n")
    if different:
        w("\nDIFFERENT analyses:\n")
        for token, lib_stems, legacy_stems in different:
            w(f"    {token:35s} library={lib_stems}  app_root={legacy_stems}\n")

    session.close()


def main():
    if '--report' in sys.argv:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            class Tee:
                def write(self, data):
                    sys.stdout.write(data)
                    f.write(data)
            report(Tee())
        print(f"\nreport written to {REPORT_PATH}")
    else:
        unittest.main()


if __name__ == '__main__':
    main()
