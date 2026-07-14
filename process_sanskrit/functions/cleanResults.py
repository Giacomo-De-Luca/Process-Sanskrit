from process_sanskrit.utils.lexicalResources import filtered_words
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES
from process_sanskrit.functions.dictionaryLookup import dict_search
import re
import regex

def extract_roots(list_of_entries, debug=False):
    roots = []
    i = 0
    
    while i < len(list_of_entries):
        current_entry = list_of_entries[i]
        
        # If entry has element at index 4 (original word)
        if len(current_entry) > 4:
            original_word = current_entry[4]
            stemmed_forms = []
            
            # Collect all stemmed forms for this original word
            j = i
            while j < len(list_of_entries) and len(list_of_entries[j]) > 4 and list_of_entries[j][4] == original_word:
                if list_of_entries[j][0] not in stemmed_forms:  # Avoid duplicates
                    stemmed_forms.append(list_of_entries[j][0])
                j += 1
            
            # Add as tuple if multiple stems, otherwise as single string
            if len(stemmed_forms) > 1:
                if debug:
                    print(f"Multiple stems for '{original_word}': {stemmed_forms}")
                roots.append(tuple(stemmed_forms))
            else:
                if debug:
                    print(stemmed_forms[0])
                roots.append(stemmed_forms[0])
                
            i = j  # Skip already processed entries
        else:
            # Handle entries without entry[4]
            if not roots or (isinstance(roots[-1], str) and roots[-1] != current_entry[0]) or \
               (isinstance(roots[-1], tuple) and current_entry[0] not in roots[-1]):
                if debug:
                    print(current_entry[0])
                roots.append(current_entry[0])
            i += 1
    
    return roots

def roots_splitted(list_of_entries, debug=False):

        root_dict = {}
        separators = r"[-—,/]"
        for entry in list_of_entries:
            if len(entry) == 7:
                components = entry[5]
            elif len(entry) == 3:
                components = entry[1]
            else:
                continue

            # Dictionary rows may omit their optional component analysis.  The
            # headword is the conservative one-part representation in that case.
            if not isinstance(components, str) or not components:
                components = entry[0]

            parts = re.split(separators, components)
            parts = [regex.sub(r'[^\p{L}]', '', part) for part in parts if part]
            parts = list(dict.fromkeys(parts))  # Remove duplicates while preserving order
            if entry[0] not in root_dict:
                root_dict[entry[0]] = parts
        return root_dict
    

## The prefixes worth re-joining, mapped to every stem root_any_word emits for the
## prefix *itself* -- the homographs it finds alongside the upasarga, which sit
## between the prefix and the real stem and have to be stepped over to reach it.
## They are not guesses: root_any_word("sam") -> sam, sa, sa (the avyaya plus the
## -a noun `sa` whose Acc.Sg is `sam`), root_any_word("ava") -> ava, ava, av (the
## avyaya plus the verb root `av`), root_any_word("anu") -> anu, anu (no twin).
##
## Getting `ava` wrong here is what the old [j + 1] index was papering over: it
## skipped one entry past the stem, which happened to hop the single `av` filler,
## so `avaruhya` re-joined as `avaruh` by accident.  The moment the filler was
## absent, or the stem sat last, the same +1 read the wrong word or ran off the
## end.  Absorb the fillers by name and the index needs no fudge.
## `duḥ` is the odd one out and belongs here anyway.  It is not an upasarga -- it
## is absent from SANSKRIT_PREFIXES and root_any_word("duḥ") is None -- so it is
## never *stripped*; it reaches the entry list only when the compound splitter cuts
## a word like duḥkha in two.  That gives it no homographs of its own and so an
## empty absorbed set, but the re-join it needs is the identical operation, and the
## hand-written duḥ/kha block that used to do it here read `list_of_entries[1 + 2]`
## (a typo for `i + 2`) into an already-shortened list and raised IndexError.
REJOINABLE_PREFIXES = {
    "sam": ("sam", "sa"),
    "anu": ("anu",),
    "ava": ("ava", "av"),
    "duḥ": (),
}


def _canonical_headword(voc_entry, queried):
    """The headword the dictionaries file this word under, or `queried` if none do.

    `dict_search` folds sam -> saṃ (samMap) for the *lookup* but echoes the query
    back at slot [0], so a re-join of `sam` + `vedana` would report the lemma
    `samvedana` while both other authorities in the pipeline say `saṃvedana` --
    the forms DB (`root_any_word("samvedana") -> saṃvedana`) and Monier-Williams,
    which files the entry under `saṃvedana`.  The payload we already hold carries
    the real headword as its inner key, so taking it costs no extra lookup.

    Self-limiting by construction: where no fold happened the headword *is* the
    query (`samādhi`, `avagraha`), and this returns it unchanged.
    """
    payload = voc_entry[2]
    for dictionary in payload.values():
        for headword in dictionary:
            return headword
    return queried


def rejoin_prefix(list_of_entries, i, absorbed):
    """Collapse a stripped prefix back into the compound it came from.

    `root_any_word` resolves `samādhi` by stripping `sam` off `ādhi`; the word is
    lexicalised, so it should reach the caller whole.  Re-joining is only correct
    when the joined form is *attested*, and that is the trap this function exists
    to hold:

    `dict_search` never returns None.  A word it cannot find comes back as a stub
    whose slot [2] is a list holding the word itself, while a real hit carries a
    dict keyed by dictionary name.  Merging on the stub replaces a correct prefix
    analysis (sam + upekṣa) with a lookup failure for a headword that does not
    exist (samupekṣa) -- which is how `samupekṣa` came back as nothing but itself.
    Only ever merge on `isinstance(..., dict)`.

    No spelling fallback belongs here: `dict_search` already folds sam -> saṃ via
    samMap, so `samyoga` finds the `saṃyoga` entry on its own.  It folds for the
    *lookup* only and echoes the query back at slot [0], so the lemma is taken from
    the payload instead -- see `_canonical_headword`.

    A stem that is itself a prefix is not a headword to look up, so stacked
    prefixes (`sam` + `ava` + ...) are deliberately out of scope and never re-join.

    Mutates `list_of_entries` in place; a no-op unless the join is attested.
    """
    prefix = list_of_entries[i][0]

    j = i + 1
    while j < len(list_of_entries) and list_of_entries[j][0] in absorbed:
        j += 1
    if j >= len(list_of_entries):
        return

    stem = list_of_entries[j][0]
    if stem in SANSKRIT_PREFIXES:
        return

    queried = prefix + stem
    voc_entry = dict_search([queried])
    if not voc_entry or len(voc_entry[0]) <= 2 or not isinstance(voc_entry[0][2], dict):
        return

    ## dict_search returns one entry per word queried, and we queried one.
    merged = voc_entry[0]
    merged[0] = _canonical_headword(merged, queried)

    list_of_entries[i] = merged
    del list_of_entries[i + 1:j + 1]


def clean_results(list_of_entries, mode="detailed", debug=False):

    i = 0
   
    #print("is it broken here?", list_of_entries)

    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        # Check if the word is in filtered_words
        if list_of_entries[i][0] in filtered_words:
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == list_of_entries[i][0]:
                del list_of_entries[i + 1]

        ## should make a rule here that does the following. 
        ## check if a word has 'indeclinable (avyaya)'
        ## if it does, check if the next word is also the same as it
        # # if it is, delete the next word.


        if len(list_of_entries[i]) >= 5 and list_of_entries[i][0][-1] == "n" and list_of_entries[i][4] != list_of_entries[i][0]:
            #print("the one not replaced:", list_of_entries[i])
            if list_of_entries[i][4] in DICTIONARY_REFERENCES:
                replacement = dict_search([list_of_entries[i][4]])
                if replacement is not None:
                    list_of_entries[i] = replacement[0]
        

        
        absorbed = REJOINABLE_PREFIXES.get(list_of_entries[i][0])
        if absorbed is not None:
            rejoin_prefix(list_of_entries, i, absorbed)

        i += 1
    

    if mode == "parts":
            return roots_splitted(list_of_entries, debug=debug)
    elif mode == "roots":
            return extract_roots(list_of_entries, debug=debug)
    else:  # Default case when roots is "none" or any other value
        return list_of_entries
