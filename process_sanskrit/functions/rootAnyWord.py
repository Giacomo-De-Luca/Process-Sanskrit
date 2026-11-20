import time
from contextlib import contextmanager
from contextvars import ContextVar
from process_sanskrit.functions.SQLiteFind import SQLite_find_name, SQLite_find_verb
from process_sanskrit.utils.loadResources import type_map
from process_sanskrit.utils.lexicalResources import variableSandhi
from process_sanskrit.utils.lexicalResources import (
    FINAL_AU_SPELLING,
    PREFIX_LOOKUP_SPELLINGS,
    SANSKRIT_PREFIXES,
    samMap,
)
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


## How many roots of a verb form the lookups report.  The verb table files 2.2%
## of its forms under several roots (`kariṣyati`: `kṝ`, `kṛ`), and
## `SQLite_find_verb` returns them all, the historical answer first.  Lemma mode
## reads them all.  `roots`, `parts` and detailed output read the first only,
## as they always have: `clean_results` re-joins a prefix with one root, so the
## others would come back as pieces of their own (`saṃkariṣyati` -> `saṃkṝ`,
## `kṛ`).  See documentation/lemma-mode.md, "Verb forms of several roots".
_EVERY_VERB_ROOT = ContextVar("every_verb_root", default=False)


@contextmanager
def every_verb_root(enabled=True):
    """Within the block, the lookups report every root of a verb form, not the first only.

    The setting belongs to the current thread or task: work handed to another
    thread does not inherit it, so hold the block where the lookups run, as
    `LemmaAnalyzer.analyze` does.
    """
    token = _EVERY_VERB_ROOT.set(enabled)
    try:
        yield
    finally:
        _EVERY_VERB_ROOT.reset(token)


def _direct_roots(word, session, memo):
    """Return direct database matches, memoized within one processing request.

    Nominal readings come first, then verbal ones.  The memo keeps every root
    of a verb form, and `every_verb_root` decides how many are returned, so one
    memo serves either setting.  A miss returns the entries as read, a hit a
    thawed copy of what the memo holds.
    """
    if word in memo:
        names, verbs = (_thaw(readings) for readings in memo[word])
    else:
        names = SQLite_find_name(word, session=session) or []
        verbs = SQLite_find_verb(word, session=session) or []
        for result in names + verbs:
            abbreviation = result[1]
            if abbreviation in type_map:
                result[1] = type_map[abbreviation]
        memo[word] = (_freeze(names), _freeze(verbs))

    if not _EVERY_VERB_ROOT.get():
        verbs = verbs[:1]
    return names + verbs or None


def _stamp_whole_word(matches, word):
    """Record `word` as the surface of the entries that resolved the remainder.

    `entry[4]` is read two incompatible ways downstream.  `extract_roots` folds a
    run of entries that share it into one tuple of rival readings, so a *prefix*
    entry must keep its own surface -- stamped with the whole word, `upa` and
    `diś` read as "upa OR diś" rather than as the two words of `upadiśyate`.  But
    the `-n` lemma rule and the whole-word dictionary check both want the whole
    word an analysis came from, and they only ever look at the root entries.

    Those are the entries sharing the *last* match's surface: the remainder's own
    lookup appends them last, while a deeper decomposition (`a` + `vi` + `rati`)
    leaves its inner prefix entries a surface of their own, which puts them
    outside the group -- re-stamping those would collapse the nested split again.

    Take that surface from the entries, never from the remainder as requested:
    the lookup may have resolved through a sandhi variant or `samMap`
    (`pratisaṃvedanā` reaches its stem as `samvedanā`), and then no entry carries
    the remainder that was asked for, so keying on it would stamp nothing at all.
    """
    root = matches[-1] if matches else None
    if not isinstance(root, list) or len(root) != 5:
        return
    span = root[4]
    for match in matches:
        if isinstance(match, list) and len(match) == 5 and match[4] == span:
            match[4] = word


def _prefix_entries(prefix, session, memo):
    """The analysis of a stripped prefix as a word of its own; [] if it has none.

    A prefix is looked up like any other word, so a spelling that is not a word
    finds nothing: `vy` is only `vi` before a vowel.  Looked up as written it
    used to drop out of the result without a trace, and `vyavaharati` came back
    as `avahṛ`.  `PREFIX_LOOKUP_SPELLINGS` names the spelling the forms table
    files each such prefix under.
    """
    spelling = PREFIX_LOOKUP_SPELLINGS.get(prefix, prefix)
    return root_any_word(spelling, session=session, _memo=memo) or []


def _record_prefix(trace, prefix, prefix_entries, matches, remainder):
    """Tell `trace` which prefix was stripped and what it left; no-op without one.

    Must run before `_stamp_whole_word`, which overwrites the very surface being
    recorded.  The prefix is named by the surface its own entries carry -- the
    spelling the forms table files it under (`prati` for a stripped `praty`) --
    and the remainder by the surface the stem was matched as.
    """
    if trace is None:
        return
    name = prefix_entries[0][4] if prefix_entries else PREFIX_LOOKUP_SPELLINGS.get(prefix, prefix)
    stem = matches[-1] if matches else None
    matched = stem[4] if isinstance(stem, list) and len(stem) == 5 else remainder
    trace.record(name, matched)


def root_any_word(
    word,
    attempted_words=None,
    timed=False,
    session=None,
    _memo=None,
    allow_prefixes=True,
    _trace=None,
):

    if word == 'api' or word == 'āpi':
        # Preserve the established particle reading in the same five-field
        # shape as database analyses. A flat stub was treated as three entries.
        return [['api', 'indeclinable (avyaya)', None, ['api'], word]]
    
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
    

    ## a final -āv is -au before a vowel: a split token `samādhāv` is `samādhau`
    ## (documentation/final-av.md); the single-word cascade reads it so already
    spelling, sound = FINAL_AU_SPELLING
    if word.endswith(spelling):
        tentative = word[:-len(spelling)] + sound
        if tentative not in attempted_words:
            attempt = root_any_word(
                tentative,
                attempted_words,
                timed,
                session=session,
                _memo=_memo,
                allow_prefixes=False,
            )
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
            _trace=_trace,
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
                _trace=_trace,
            )
            if attempt is not None:
                prefix_entries = _prefix_entries(prefix, session, _memo)
                _record_prefix(_trace, prefix, prefix_entries, attempt, remainder)
                result = prefix_entries + attempt
                _stamp_whole_word(attempt, word)
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
                            outer_entries = _prefix_entries(prefix, session, _memo)
                            inner_entries = _prefix_entries(nested_prefix, session, _memo)
                            _record_prefix(
                                _trace, nested_prefix, inner_entries, nested_attempt, nested_remainder
                            )
                            _record_prefix(_trace, prefix, outer_entries, nested_attempt, remainder)
                            result = outer_entries + inner_entries + nested_attempt
                            _stamp_whole_word(nested_attempt, word)
                            return result
            
    return None
