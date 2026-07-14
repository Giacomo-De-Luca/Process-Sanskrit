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
    sanskritFixedSandhiMap, 
    SANSKRIT_PREFIXES
)
from process_sanskrit.utils.transliterationUtils import (
    transliterate,
    normalize_avagraha,
    restore_avagraha,
)

### import the sandhiSplitScorer and construct the scorer object. 

from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.dictionaryLookup import (
    DEFAULT_DICTIONARY,
    consult_references,
    dict_search,
    multidict,
)
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.taddhitaDerivation import taddhita_deriver
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.databaseSetup import session_scope, with_session, requires_database



### get the version of the library

def preprocess(text, max_length=150, debug=False):

    ## editions, OCR and PDF copy-paste each spell the avagraha with a different
    ## glyph ('  ’  ʼ  `  ´ ...); fold them onto the ASCII apostrophe before
    ## scheme detection or the \p{L} filter below ever sees them
    text = normalize_avagraha(text)

    text = transliterate(text, "IAST")

    ## if the text is too long, we try to trim it to the last whitespace
    if len(text) > max_length:
        last_space_index = text[:max_length].rfind(' ')
        if last_space_index == -1:
            text = text[:max_length]
        else:
            # Trim up to the last whitespace
            text = text[:last_space_index]

    ## TODO 
    ## this may lead to errors
    ## it should be like this:
    ## if jj in text
    ## check if jj occours inside one of the 20 words or so that have jj inside
    ## in that case keep it,
    ## otherwise replace it with j j
    ## this is a temporary fix, it should be improved
    if 'jj' in text:
        text = text.replace('jj', 'j j')

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
            added to `forwarded` below.

    Returns:
        List containing processed entries if special handling occurred,
        None if no special handling was needed
    """
    forwarded = dict(
        max_length=max_length,
        debug=debug,
        mode=mode,
        session=session,
        _memo=_memo,
        cached=cached,
    )
    dict_names = dict_names or ()

    ## both wildcard branches have two exits: a hit returns the dictionary entry
    ## directly, a miss falls back to processing the text without the wildcard.
    ## The hit used to bypass clean_results, so it ignored `mode` -- a roots
    ## request came back as detailed entries.  For a pattern (_ or %) the stem is
    ## the pattern itself; a pattern has no root to speak of.

    # Handle wildcard search with asterisk
    if text.endswith('*'):
        transliterated_text = transliterate(text[:-1], "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        if not isinstance(voc_entry[0][2], list):
            return clean_results(voc_entry, debug=debug, mode=mode)
        return process(text[:-1], *dict_names, **forwarded)

    # Handle explicit wildcard search with _ or %
    if '_' in text or '%' in text:
        transliterated_text = transliterate(text, "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        if not isinstance(voc_entry[0][2], list):
            return clean_results(voc_entry, debug=debug, mode=mode)
        return process(text, *dict_names, **forwarded)

    # Handle pre-split compounds with - or +
    if "-" in text or "+" in text:
        word_list = re.split(r'[-+]', text)
        processed_results = []
        for word in word_list:
            ## a leading, trailing or doubled separator splits to an empty
            ## segment, which carries no analysis to contribute
            if not word:
                continue
            result = process(word, *dict_names, **forwarded)
            processed_results.extend(result)
        return processed_results

    return None  # Return None if no special cases matched


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
):

    raw_text = text
    text = preprocess(text, max_length=max_length, debug=debug)
    request_memo = {} if _memo is None else _memo

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

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        
        if text and text[-1] in sanskritFixedSandhiMap:
            text = text[:-1] + sanskritFixedSandhiMap[text[-1]]

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text, session=session, _memo=request_memo)
        if debug == True:
            print("rooting result", result)

        if result is None and "ṅ" in text:
            ## this is removed, it was not triggering, and it was not clear if it was useful: or "ñ" in text
            tentative = text.replace("ṅ", "ṃ")
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo
            )
            if attempt is not None:
                result = attempt
        
        if result is None and "ṁ" in text:
            tentative = text.replace("ṁ", "ṃ")
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo
            )
            if attempt is not None:
                result = attempt

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "ch":
            #print("tentative", text)
            tentative = 'ś' + text[1:] 
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo
            )
            #print("attempt", attempt)
            if attempt is not None:
                result = attempt

        if result is None:
            # Check if word exists in dictionary references before trying to split
            if text in DICTIONARY_REFERENCES:
                if debug:
                    print(f"Found {text} in dictionary references, doing direct lookup")
                result_vocabulary = dict_search([text], *dict_names, session=session)
                if isinstance(result_vocabulary[0][2], dict):
                    return clean_results(result_vocabulary, debug=debug, mode=mode)

        if result is not None:
            if debug == True: 
                print("Getting some results with no splitting here:", result)
            for i, res in enumerate(result):
                if isinstance(res, str):
                    result[i] = res.replace('-', '')
                elif isinstance(res, list):
                    if isinstance(res[0], str):
                        res[0] = res[0].replace('-', '')
            result_vocabulary = dict_search(result, *dict_names, session=session)

            if debug == True: 
                print("result_vocabulary", result_vocabulary)

            ## TODO the following employs a wrong logic and should be edited
            ## we should add the dictionary entry as a possibility only instead  
            ## and attach it to the list, giving it a from : 'original entry'
            ## also it should Never check for the final 'H'. otherwise it will trigger all the time using the APTE dict 
            ## in case of nominatives.

            ## if the word is inside the dictionary, we return the entry directly, since it will be accurate.
            ## 
            if isinstance(result_vocabulary, list):

                ## Ask the *last* entry for the whole word, not result[0]: when the
                ## word was resolved by stripping a prefix, result[0] is the prefix
                ## (`vi` of `virati`) and carries only its own surface, so keying the
                ## lookup on it drops the attested entry for the whole word.  The
                ## root entries carry the whole word *as matched*, which is also why
                ## the raw `text` will not do -- saṃskāro is matched as saṃskāraḥ,
                ## saṅgrahaḥ as saṃgrahaḥ, and only the matched form is a headword.
                last_entry = result[-1]
                whole_word = (
                    last_entry[4]
                    if isinstance(last_entry, list) and len(last_entry) > 4
                    else None
                )
                analysed_stems = {
                    entry[0] for entry in result if isinstance(entry, list) and entry
                }
                ## Offer it only when no analysis already yields it; otherwise
                ## pratyakṣam is reported twice -- once as itself, once as the
                ## inserted headword.
                if (
                    whole_word
                    and whole_word not in analysed_stems
                    and whole_word in DICTIONARY_REFERENCES
                ):
                    replacement = dict_search([whole_word], *dict_names, session=session)
                    if debug:
                        print("replacement", replacement[0])
                        print("len replacement", len(replacement[0]))
                    ## A miss is a stub, never None -- see documentation/prefix-rejoin.md
                    if replacement and isinstance(replacement[0][2], dict):
                        result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary, debug=debug, mode=mode)
        else:
            ## if result is None, we try to find the word in the dictionary for exact match
            result_vocabulary = dict_search([text], *dict_names, session=session)
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary, debug=debug, mode=mode)

            ## The lexicon lists only the *lexicalised* -tā / -tva abstract nouns.
            ## Both suffixes are productive, so a coined derivative (niṣyanda-tā)
            ## reaches this point unresolved -- and the splitter below would cut
            ## it in two and read the orphaned suffix as a verb (tā -> tṛ, tan),
            ## losing the lemma.  Rebuild it from its base instead.  This runs
            ## last on purpose: an attested word, however it ends, is already
            ## resolved above and never gets here.
            derived = taddhita_deriver.derive(text, session=session)
            if derived is not None:
                if debug:
                    print("taddhita derivation", derived)
                ## Gloss the *base*, not the derivative: niṣyandatā heads no
                ## dictionary entry, but niṣyanda does, and that is where its
                ## meaning lives.  Appending the payload here mirrors what
                ## dict_search does for an attested word, and yields the same
                ## entry shape -- the asymmetry (lemma from the derivation,
                ## gloss from the base) is why dict_search cannot do it for us.
                entry = derived.as_entry() + consult_references(
                    derived.base, *(dict_names or (DEFAULT_DICTIONARY,)), session=session
                )
                return clean_results([entry], debug=debug, mode=mode)

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

    ## should this really be kept? 
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]
      
    return clean_results(inflections_vocabulary, debug=debug, mode=mode)
