"""The single-word cascade: forms table, dictionary, -tā/-tva deriver.

`process()` resolves a single word through a fixed sequence before it falls back
to splitting it.  The order is load-bearing -- the deriver must come after the
whole-word dictionary lookup, or a manufactured analysis outranks an attested
word (vārtā -> vār + tā) -- so it lives here once, and both the older modes and
`mode="lemmas"` follow it.

The body of `resolve` was moved out of `process()` unchanged.  What it used to
hand straight to `clean_results` it now returns as a `WordResolution`, which
also says which stage resolved the word, which entry is the inserted whole-word
headword, and which prefixes were stripped.
"""

import re
from dataclasses import dataclass
from typing import Optional

from process_sanskrit.functions.dictionaryLookup import (
    DEFAULT_DICTIONARY,
    consult_references,
    dict_search,
)
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.taddhitaDerivation import taddhita_deriver
from process_sanskrit.utils.analysisTrace import PrefixTrace
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import read_final_sandhi

## the characters a one-word query may use as a dictionary pattern
WILDCARD_CHARACTERS = re.compile(r"[*_%]")


@dataclass
class WordResolution:
    """How far the cascade got with one word.

    `text` is the word as analysed: the cascade reads a final -y, -r or -v back
    as the sound it stands for, and whoever splits the word next must use that.
    `entries` is the list the older modes render, ready for `clean_results`;
    it is empty when the stage is UNRESOLVED.  `headword` is the whole-word
    dictionary entry the cascade put in front of `entries`, when the surface
    form is itself a headword (`gacchati` in the Pali dictionary); it is also
    `entries[0]`.  `prefixes` is what the forms-table lookup stripped.
    """

    MORPHOLOGY = "morphology"
    DICTIONARY = "dictionary"
    DERIVED = "derived"
    UNRESOLVED = "unresolved"

    text: str
    stage: str
    entries: list
    headword: Optional[list]
    prefixes: PrefixTrace

    @property
    def resolved(self) -> bool:
        return self.stage != self.UNRESOLVED


class WordCascade:
    """Resolve one word without splitting it, or say that it cannot be done."""

    ## `jj` is a double consonant of the word (`lajjā`, `rajju`, `ujjvala`) or
    ## the sandhi of a final t/d before j (`tat` + `japaḥ` -> `tajjapaḥ`), which
    ## the splitter does not undo.  See `read_double_j`.
    DOUBLE_J = "jj"
    DOUBLE_J_CUT = "j j"
    _SEPARATOR = re.compile(r"([-+])")

    def __init__(self, dict_names=(), *, session, memo, debug=False):
        self.dict_names = tuple(dict_names or ())
        self.session = session
        self.memo = memo
        self.debug = debug

    def read_double_j(self, text):
        """`text` with the `jj` of each word cut as `j j`, unless the word resolves whole.

        Text cleaning used to cut every `jj`, so `lajjā` reached the lookups as
        `laj` + `jā`.  A word, or a segment of a pre-split word, keeps its `jj`
        when the cascade resolves it as it stands; one that does not is cut as
        it always was, and `taj` then reaches `tad` through the final-`j`
        sandhi variant.  A one-word pattern is a headword query and is left as
        typed; in running text a pattern character makes no pattern, and the
        word is read like any other.  Text without `jj` is returned unread.

        The decision is taken on a memo of its own.  The request memo hands a
        word it holds back as a copy whose tags are lists, where a fresh lookup
        gives tuples, so deciding on it would change the type of the tags the
        analysis of the same word reports.
        """
        if self.DOUBLE_J not in text:
            return text
        words = text.split(" ")
        if len(words) == 1 and WILDCARD_CHARACTERS.search(text):
            return text
        decider = WordCascade(self.dict_names, session=self.session, memo={})
        return " ".join(decider._read_double_j_word(word) for word in words)

    def _read_double_j_word(self, word):
        if self.DOUBLE_J not in word:
            return word
        return "".join(
            segment
            if self.DOUBLE_J not in segment or self.resolve(segment).resolved
            else segment.replace(self.DOUBLE_J, self.DOUBLE_J_CUT)
            for segment in self._SEPARATOR.split(word)
        )

    def resolve(self, text):
        session, request_memo, debug, dict_names = (
            self.session, self.memo, self.debug, self.dict_names,
        )
        trace = PrefixTrace()

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        
        text = read_final_sandhi(text)

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text, session=session, _memo=request_memo, _trace=trace)
        if debug == True:
            print("rooting result", result)

        if result is None and "ṅ" in text:
            ## this is removed, it was not triggering, and it was not clear if it was useful: or "ñ" in text
            tentative = text.replace("ṅ", "ṃ")
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo, _trace=trace
            )
            if attempt is not None:
                result = attempt
        
        if result is None and "ṁ" in text:
            tentative = text.replace("ṁ", "ṃ")
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo, _trace=trace
            )
            if attempt is not None:
                result = attempt

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "ch":
            #print("tentative", text)
            tentative = 'ś' + text[1:] 
            attempt = root_any_word(
                tentative, session=session, _memo=request_memo, _trace=trace
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
                    return WordResolution(text, WordResolution.DICTIONARY, result_vocabulary, None, trace)

        if result is not None:
            headword = None
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
                        headword = replacement[0]

            #print("result_vocabulary", result_vocabulary)
            return WordResolution(text, WordResolution.MORPHOLOGY, result_vocabulary, headword, trace)
        else:
            ## if result is None, we try to find the word in the dictionary for exact match
            result_vocabulary = dict_search([text], *dict_names, session=session)
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return WordResolution(text, WordResolution.DICTIONARY, result_vocabulary, None, trace)

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
                return WordResolution(text, WordResolution.DERIVED, [entry], None, trace)

        return WordResolution(text, WordResolution.UNRESOLVED, [], None, trace)
