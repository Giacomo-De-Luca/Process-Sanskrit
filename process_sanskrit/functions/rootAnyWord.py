import time 
from process_sanskrit.functions.SQLiteFind import SQLite_find_name, SQLite_find_verb
from process_sanskrit.utils.loadResources import type_map
from process_sanskrit.utils.lexicalResources import variableSandhi
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES, samMap
from dataclasses import dataclass
from typing import Optional, List
 

##given a name finds the root


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _direct_roots(word, session, memo):
    """Return direct database matches, memoized within one processing request."""
    if word in memo:
        return _thaw(memo[word])

    result_roots_name = SQLite_find_name(word, session=session)
    result_roots_verb = SQLite_find_verb(word, session=session)

    if result_roots_name and result_roots_verb:
        result_roots = result_roots_name + result_roots_verb
    elif result_roots_name:
        result_roots = result_roots_name
    elif result_roots_verb:
        result_roots = result_roots_verb
    else:
        result_roots = None

    if result_roots:
        for result in result_roots:
            abbreviation = result[1]
            if abbreviation in type_map:
                result[1] = type_map[abbreviation]

    memo[word] = _freeze(result_roots)
    return result_roots


def _stamp_whole_word(matches, span, word):
    """Record `word` as the surface of the entries that resolved `span` itself.

    `entry[4]` is read two incompatible ways downstream.  `extract_roots` folds a
    run of entries that share it into one tuple of rival readings, so a *prefix*
    entry must keep its own surface -- stamped with the whole word, `upa` and
    `diś` read as "upa OR diś" rather than as the two words of `upadiśyate`.  But
    the `-n` lemma rule and the whole-word dictionary check both want the whole
    word an analysis came from, and they only ever look at the root entries.
    Stamping exactly the entries that resolved `span` serves both.

    `matches` can already hold a deeper decomposition (`a` + `vi` + `rati`), whose
    inner prefix entries carry their own surface and so fail the `span` test --
    which is the point: re-stamping them would collapse the nested split again.
    """
    for match in matches:
        if len(match) == 5 and match[4] == span:
            match[4] = word


def root_any_word(
    word,
    attempted_words=None,
    timed=False,
    session=None,
    _memo=None,
    allow_prefixes=True,
):

    if word == 'api' or word == 'āpi':
        return ['api' , 'api' , ['api']]  
    
    if attempted_words is None:
        attempted_words = frozenset()
    if _memo is None:
        _memo = {}
    
    result_roots = None

    # If the word has already been attempted, return None to avoid infinite loop
    if word in attempted_words:
        return None

    # Create a new frozenset with the current word added
    attempted_words = frozenset([word]).union(attempted_words)

    if timed:
        start_time = time.time()

    if word:
        result_roots = _direct_roots(word, session, _memo)
    else:
        return None
    
    if timed:
        print(f"SQLite_find_name({word}) took {time.time() - start_time:.6f} seconds")


    if result_roots:
        return result_roots

    # If no result is found, try replacements based on variableSandhi
    ## prefix stripping is deferred inside the variant subtree: a prefix split
    ## found deep in one variant must not preempt a whole-word match reachable
    ## through a later variant (e.g. utkrāntiś -> utkrāntiḥ -> utkrānti,
    ## not ut + krānti)
    if word[-1] in variableSandhi:
        for replacement in variableSandhi[word[-1]]:
            tentative = word[:-1] + replacement
            if timed:
                start_time = time.time()
            if tentative not in attempted_words:
                #print (f"tentative: {tentative}")
                #print (f"attempted_words: {attempted_words}")
                attempt = root_any_word(
                    tentative,
                    attempted_words,
                    timed,
                    session=session,
                    _memo=_memo,
                    allow_prefixes=False,
                )
                if timed:
                    print(f"root_any_word({tentative}) took {time.time() - start_time:.6f} seconds")
                if attempt:
                    return attempt
    

    ##probably add a rule that if ṅ is in the word, change it with ṃ to account if for different spellings

    # Different spellings for sam, - it is so common that it deserves its own rule
    if word[0:3] in samMap:
        tentative = samMap[word[0:3]] + word[3:]
        if timed:
            start_time = time.time()
        attempt = root_any_word(
            tentative,
            attempted_words,
            timed,
            session=session,
            _memo=_memo,
            allow_prefixes=allow_prefixes,
        )
        if timed:
            print(f"root_any_word({tentative}) took {time.time() - start_time:.6f} seconds")
        if attempt is not None:
            return attempt
        
    ## Productive -tā / -tva derivatives (niṣyanda-tā) are NOT reconstructed here.
    ## They are handled in process(), after the whole-word dictionary lookup has
    ## also missed -- see functions/taddhitaDerivation.py.  Doing it at this level
    ## would let a manufactured analysis outrank an attested word: root_any_word
    ## runs before the dictionary, so vārtā would come back as vār + tā rather
    ## than as itself.

    if not allow_prefixes:
        return None

    for prefix in SANSKRIT_PREFIXES:
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            attempt = root_any_word(
                remainder,
                session=session,
                _memo=_memo,
            )
            if attempt is not None:
                if prefix == 'ud': 
                    prefix_result = root_any_word(
                        'ut', session=session, _memo=_memo
                    )
                    result = (prefix_result or []) + attempt
                else: 
                    prefix_root = root_any_word(
                        prefix, session=session, _memo=_memo
                    )
                    result = prefix_root + attempt if prefix_root else attempt
                _stamp_whole_word(attempt, remainder, word)
                return result
            else: 
                for nested_prefix in SANSKRIT_PREFIXES:
                    if remainder.startswith(nested_prefix):
                        nested_remainder = remainder[len(nested_prefix):]
                        nested_attempt = root_any_word(
                            nested_remainder,
                            session=session,
                            _memo=_memo,
                        )
                        if nested_attempt is not None:
                            prefix_result = root_any_word(
                                prefix, session=session, _memo=_memo
                            ) or []
                            nested_prefix_result = root_any_word(
                                nested_prefix, session=session, _memo=_memo
                            ) or []
                            result = prefix_result + nested_prefix_result + nested_attempt
                            _stamp_whole_word(nested_attempt, nested_remainder, word)
                            return result
            
    return None
