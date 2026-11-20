### procesSanskrit library. The goal of the library is to provide the processing of Sanskrit text in a simple and efficient way.
### The library is built on top of the SanskritParser library and the IndicTransliteration library.
### The library provides the following functionalities:
### - Sandhi splitting
### - Transliteration
### - Root extraction
### - Inflection table generation
### - Stopwords removal
### - Sandhi splitting with detailed output, multiple attempts, scoring, and caching
### - Enhanced sandhi splitting with detailed output, multiple attempts, scoring, and caching
### - Compound splitting with detailed output, multiple attempts, scoring, and caching
### - Vocabulary voice extraction from multiple dictionaries and wildcard search
### - Cleanup of the results from the previous functions
### 
### - MAIN FUNCTION:
### - Process function, executing all of the above at once
### - Return the results in a structured format
### - call process with mode='roots' to get only the root of all the words in a Sanskrit text. 


### packages and local modules import 


import re
import regex
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Dict, Tuple, Union, Optional
import time


from process_sanskrit.utils.lexicalResources import (
    variableSandhi, 
    SANSKRIT_PREFIXES
)
from process_sanskrit.utils.transliterationUtils import (
    transliterate,
    normalize_avagraha,
    restore_avagraha,
)

### import the sandhiSplitScorer and construct the scorer object. 

from process_sanskrit.functions.dictionaryLookup import (
    dict_search,
    multidict,
)
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.wordCascade import WILDCARD_CHARACTERS, WordCascade
from process_sanskrit.utils.databaseSetup import session_scope, with_session, requires_database
from process_sanskrit.utils.paradigm import is_one_letter_root



### get the version of the library

## WILDCARD_CHARACTERS (imported above) are the characters a one-word query may
## use as a dictionary pattern; they live in wordCascade.py, which reads them too


def to_iast(text, input_scheme=None):
    """The text in IAST, with every avagraha glyph folded onto the apostrophe.

    This is the text `max_length` measures, and the one `mode="lemmas"` cuts
    into tokens; `clean_iast` does the rest of what `preprocess` does.
    """
    ## editions, OCR and PDF copy-paste each spell the avagraha with a different
    ## glyph ('  ’  ʼ  `  ´ ...); fold them onto the ASCII apostrophe before
    ## scheme detection or the \p{L} filter below ever sees them
    text = normalize_avagraha(text)

    ## input_scheme=None auto-detects, which misreads short ASCII input
    ## (SLP1 "Darma" as HK "ḍarma"); callers that know the scheme pass it
    return transliterate(text, "IAST", input_scheme)


def clean_iast(text):
    """IAST text reduced to the words the analysis reads, separated by one space."""

    ## The text is NOT cut to a maximum length here.  It used to be, silently:
    ## a passage lost every word past the first hundred characters.  A long
    ## text is analysed word by word instead -- see process() and
    ## documentation/long-input.md.

    ## A `jj` is NOT cut here.  It used to be, in every word: `lajjā` reached
    ## the lookups as `laj` + `jā`.  Whether a `jj` is the word's own or the
    ## sandhi of t/d + j takes a lookup, so it is decided by
    ## WordCascade.read_double_j -- see documentation/double-j.md.

    ## restore every elided initial a-, on any word rather than only the first:
    ## the check used to be `text[0] == "'"`, so in "tasmāt so 'nupalambhena" the
    ## avagraha was left standing, then stripped as punctuation, and the word
    ## shattered ("nupalambhena" -> nu + pa + pa + lambha).  An empty string also
    ## reaches here from a bare separator ("hetu-" splits into ("hetu", "")),
    ## which the regexes tolerate where indexing did not.
    text = restore_avagraha(text)

    text = regex.sub(r"[^\p{L}'_%*+\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def preprocess(text, *, debug=False, input_scheme=None):
    """The words of `text`, in IAST, as text cleaning leaves them.

    A `jj` is still whole here: whether the analysis cuts it is decided later,
    by a lookup (`WordCascade.read_double_j`).

    The options are keyword-only: this function used to take `max_length` as
    its second argument, and a caller still passing one positionally must fail
    rather than have it read as something else.
    """
    return clean_iast(to_iast(text, input_scheme))


def _forwarded_options(*, max_length, debug, mode, session, _memo, cached):
    """The options a recursive `process()` call must be handed.

    Every such call re-enters the public entry point, so an option left out here
    is silently demoted to its default.  Anything added to `process()` must be
    added here -- with the one exception of `input_scheme`, which is pinned to
    "IAST": the text has already been converted by `preprocess()`, and reading
    it in the caller's scheme again corrupts it (IAST "dharma" read as WX is
    "ḍharma").
    """
    return dict(
        max_length=max_length,
        debug=debug,
        mode=mode,
        session=session,
        _memo=_memo,
        cached=cached,
        ## the text is IAST by now, whatever scheme the caller used
        input_scheme="IAST",
    )


def handle_special_characters(
    text: str,
    dict_names: Optional[Tuple[str, ...]] = None,
    *,
    session,
    _memo,
    cached,
    max_length,
    debug,
    mode,
) -> Optional[List]:
    """
    Handle text preprocessing for special characters including wildcards and compound splits.
    This function processes special characters that require specific handling before 
    main Sanskrit text processing can occur.
    
    The function handles three main cases:
    1. Wildcard endings with asterisk (*)
    2. Explicit wildcards using underscore (_) or percent (%)
    3. Pre-split compounds using hyphen (-) or plus (+)
    
    Args:
        text: The Sanskrit text to process
        dict_names: Optional tuple of dictionary names to search in
        session, _memo, cached, max_length, debug, mode: forwarded verbatim to
            the recursive process() calls below.  Those calls re-enter the
            public entry point, so an option dropped here is silently demoted to
            process()'s default -- which is how mode="roots" used to come back
            as detailed entries.  They are deliberately keyword-only and have no
            defaults: the defaults live on process() alone, so the two
            signatures cannot drift apart.  Anything added to process() must be
            added to `_forwarded_options`, which also explains why input_scheme
            is pinned to "IAST" rather than forwarded.

    Returns:
        List containing processed entries if special handling occurred,
        None if no special handling was needed
    """
    forwarded = _forwarded_options(
        max_length=max_length,
        debug=debug,
        mode=mode,
        session=session,
        _memo=_memo,
        cached=cached,
    )
    dict_names = dict_names or ()

    ## A trailing-star miss falls back to processing the literal word. SQL
    ## patterns return either definitions or a stub for the original pattern.
    ## The hit used to bypass clean_results, so it ignored `mode` -- a roots
    ## request came back as detailed entries.  For a pattern (_ or %) the stem is
    ## the pattern itself; a pattern has no root to speak of.

    # Handle wildcard search with asterisk
    if text.endswith('*'):
        transliterated_text = transliterate(text[:-1], "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        if not isinstance(voc_entry[0][2], list):
            return clean_results(voc_entry, debug=debug, mode=mode, dict_names=dict_names, session=session)
        return process(text[:-1], *dict_names, **forwarded)

    # Handle explicit wildcard search with _ or %
    if '_' in text or '%' in text:
        transliterated_text = transliterate(text, "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        # A pattern miss is still a valid lookup result. Reprocessing the same
        # wildcard would recurse forever once empty payloads are proper misses.
        return clean_results(voc_entry, debug=debug, mode=mode, dict_names=dict_names, session=session)

    # Handle pre-split compounds with - or +
    if "-" in text or "+" in text:
        return _process_each(re.split(r'[-+]', text), dict_names, forwarded)

    return None  # Return None if no special cases matched


def _process_each(words, dict_names, forwarded):
    """Analyse each word on its own and combine the results into one answer.

    Shared by the two callers that hand `process()` several words to analyse
    separately: the segments of a pre-split compound, and the words of a text too
    long to be analysed as one unit.  Results are concatenated, or merged for
    `parts`, whose result is a mapping.
    """
    as_mapping = forwarded["mode"] == "parts"
    combined = {} if as_mapping else []
    for word in words:
        ## a leading, trailing or doubled separator splits to an empty
        ## segment, which carries no analysis to contribute
        if not word:
            continue
        result = process(word, *dict_names, **forwarded)
        if as_mapping:
            combined.update(result)
        else:
            combined.extend(result)
    return combined


### roots should be replaced by output="roots" in the function signature
### by default, output = "detailed"
@requires_database
@with_session
def process(
    text,
    *dict_names,
    max_length=100,
    debug=False,
    mode="detailed",
    session=None,
    _memo=None,
    cached: Optional[bool] = None,
    input_scheme: Optional[str] = None,
):

    ## mode="lemmas" has its own shape and its own unit of work: one item per
    ## whitespace token of the input, each analysed as a single word.  It reports
    ## no definitions, so the dictionary names do not apply to it.
    if mode == "lemmas":
        from process_sanskrit.functions.lemmaAnalysis import LemmaAnalyzer

        return LemmaAnalyzer(
            session=session, memo=_memo, cached=cached, debug=debug
        ).analyze(text, input_scheme=input_scheme)

    raw_text = text
    iast_text = to_iast(text, input_scheme)
    text = clean_iast(iast_text)
    request_memo = {} if _memo is None else _memo
    cascade = WordCascade(dict_names, session=session, memo=request_memo, debug=debug)
    text = cascade.read_double_j(text)

    ## if text is none return empty list
    if not text:
        if mode == "roots":
            return ""
        else:
            return []

    ## the pre-split branch below is reachable only for single-word input, so a
    ## separator inside a sentence would otherwise reach the splitter verbatim --
    ## and the splitter does not read '-' or '+' as a boundary, so it merges the
    ## very segments the caller separated.  Whitespace it does honour, so demote
    ## the separator to a space and let the per-word path do the rest.
    if ' ' in text and ('-' in text or '+' in text):
        text = re.sub(r'[-+]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

    ## `max_length` is the longest text analysed as one unit.  A longer text is
    ## never cut: it is analysed word by word and the results combined, so every
    ## word of the input is in the answer.  A single word has nothing to fall
    ## back to and is analysed whole, whatever its length.
    ##
    ## The length measured is that of the text before cleaning, which is what
    ## the old cut measured: a text that was short enough to be analysed whole
    ## still is.  Wildcard characters mean something only in a one-word query;
    ## in running text they were never patterns, and must not become patterns
    ## because the words are now analysed one at a time (a stray `%` would
    ## match every headword).
    if ' ' in text and len(iast_text) > max_length:
        return _process_each(
            [WILDCARD_CHARACTERS.sub('', word) for word in text.split(' ')],
            dict_names,
            _forwarded_options(
                max_length=max_length,
                debug=debug,
                mode=mode,
                session=session,
                _memo=request_memo,
                cached=cached,
            ),
        )

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        check_special_characters = handle_special_characters(
            text,
            dict_names,
            session=session,
            _memo=request_memo,
            cached=cached,
            max_length=max_length,
            debug=debug,
            mode=mode,
        )
        if check_special_characters is not None:
            return check_special_characters

        ## The single-word cascade: forms table, dictionary, -tā/-tva deriver.  It
        ## lives in WordCascade so that mode="lemmas" follows the same order.
        resolution = cascade.resolve(text)
        text = resolution.text
        if resolution.resolved:
            return clean_results(resolution.entries, debug=debug, mode=mode, dict_names=dict_names, session=session)

    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case


    from process_sanskrit.functions.hybridSplitter import analyze_hybrid
    from process_sanskrit.functions.inflect import inflect

    from process_sanskrit.utils.analysisCache import (
        ANALYSIS_ALGORITHM_VERSION,
        CacheKey,
        CacheRecord,
        get_analysis_cache,
        lexicon_fingerprint,
        resolve_cache_enabled,
    )

    cache = None
    cache_key = None
    cached_record = None
    if resolve_cache_enabled(cached):
        cache = get_analysis_cache(force_enabled=cached is True)
        cache_key = CacheKey.from_settings(
            normalized_input=text,
            analysis_kind="hybrid_morphology",
            algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
            lexicon_fingerprint=lexicon_fingerprint(),
            settings={"attempts": 20, "score_threshold": 0.535},
        )
        cached_record = cache.get(cache_key)

    if cached_record is not None and cached_record.grammar is not None:
        splitted_text = list(cached_record.split)
        inflections = cached_record.grammar
    else:
        lock_context = (
            cache.compute_lock(cache_key)
            if cache is not None and cache_key is not None
            else None
        )
        if lock_context is None:
            analysis_started = time.perf_counter()
            analysis = analyze_hybrid(
                text, session=session, _memo=request_memo
            )
            splitted_text = analysis.split
            inflections = inflect(
                splitted_text, session=session, _memo=request_memo
            )
        else:
            with lock_context as acquired:
                if acquired:
                    cached_record = cache.get(cache_key)
                if acquired and cached_record is not None and cached_record.grammar is not None:
                    splitted_text = list(cached_record.split)
                    inflections = cached_record.grammar
                else:
                    analysis_started = time.perf_counter()
                    analysis = analyze_hybrid(
                        text, session=session, _memo=request_memo
                    )
                    splitted_text = analysis.split
                    inflections = inflect(
                        splitted_text, session=session, _memo=request_memo
                    )
                    if acquired:
                        canonical = cache.store(
                            CacheRecord(
                                key=cache_key,
                                raw_input=str(raw_text),
                                split=splitted_text,
                                grammar=inflections,
                                score=analysis.score,
                                subscores=analysis.subscores,
                                result_source=analysis.source,
                                status=analysis.status,
                                compute_ms=(
                                    time.perf_counter() - analysis_started
                                )
                                * 1000,
                            )
                        )
                        if canonical.grammar is not None:
                            splitted_text = list(canonical.split)
                            inflections = canonical.grammar
    if debug == True:
        print("splitted_text_here", splitted_text)
    if debug == True:
        print("inflections after splitting", inflections)
    inflections_vocabulary = dict_search(inflections, *dict_names, session=session)

    ## One-letter lemmas are scanner noise: root_compounds walks an unresolved
    ## token a character at a time, and every single vowel is a headword.  The
    ## verb roots `i` and `ṛ` are the exception -- filtering on length alone cost
    ## `tatra eti` its verb.  Inflector has already withheld the ones the scanner
    ## cut out of a larger token, so a one-letter root that reaches this point
    ## resolved a whole token.
    inflections_vocabulary = [
        entry for entry in inflections_vocabulary
        if len(entry[0]) > 1 or is_one_letter_root(entry)
    ]
      
    return clean_results(inflections_vocabulary, debug=debug, mode=mode, dict_names=dict_names, session=session)
