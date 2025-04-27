import time 
from process_sanskrit.functions.SQLiteFind import SQLite_find_name, SQLite_find_verb
from process_sanskrit.utils.loadResources import type_map
from process_sanskrit.utils.lexicalResources import variableSandhi
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES, samMap
from dataclasses import dataclass
from typing import Optional, List
from functools import lru_cache
 

##given a name finds the root


@dataclass
class TvaAnalysis:
    """
    Represents how to handle a tva-suffixed form by specifying what ending
    needs to be attached to the base word for analysis.
    
    Attributes:
        ending: The ending to attach to the base word after removing tva part
    """
    ending: str

def handle_tva(word: str) -> Optional[List]:
    """
    Analyzes words containing the -tva suffix by reconstructing the base word
    with appropriate endings for root analysis.
    
    Example:
        For śūnyatvānām:
        1. Identifies -tvānām ending
        2. Gets base śūnya
        3. Analyzes śūnyānām through root_any_word
    """
    # Dictionary mapping tva forms to their analysis ending
    tva_paradigm = {
        # Singular
        'tvam': TvaAnalysis('am'),      # nom/acc: śūnyam
        'tvena': TvaAnalysis('ena'),    # inst: śūnyena 
        'tvāya': TvaAnalysis('āya'),    # dat: śūnyāya
        'tvāt': TvaAnalysis('āt'),      # abl: śūnyāt
        'tvasya': TvaAnalysis('asya'),  # gen: śūnyasya
        'tve': TvaAnalysis('e'),        # loc: śūnye
        
        # Dual
        'tvābhyām': TvaAnalysis('ābhyām'),  # śūnyābhyām
        'tvayoḥ': TvaAnalysis('ayoḥ'),      # śūnyayoḥ
        
        # Plural
        'tvāni': TvaAnalysis('āni'),      # śūnyāni
        'tvebhyaḥ': TvaAnalysis('ebhyaḥ'), # śūnyebhyaḥ
        'tvānām': TvaAnalysis('ānām'),     # śūnyānām
        'tveṣu': TvaAnalysis('eṣu'),       # śūnyeṣu
        
        # Base forms 
        'tva': TvaAnalysis(''),     # Just analyze base
        'tvā': TvaAnalysis('')      # Just analyze base
    }

    if word == "tva":
        return None

    # Try to match a tva ending
    for tva_form, analysis in sorted(tva_paradigm.items(), key=lambda x: len(x[0]), reverse=True):
        if word.endswith(tva_form):
            # Get the base by removing tva form
            base = word[:-len(tva_form)]
            
            print(f"Found tva form: {tva_form}, base: {base}")
            if base[-1] == 'a':

                # Create the form to analyze by adding the appropriate ending
                analysis_form = base[:-1] + analysis.ending

                print(f"Reconstructed form: {analysis_form}")
                
                # Analyze this reconstructed form
                base_analysis = root_any_word(analysis_form, session=session)

                print(f"Base analysis: {base_analysis}")
            
            else: 
                base_analysis = root_any_word(base, session=session)
                ## here it should replace the case ending with the corrisponding case ending of the tva form
                ## base analysis[2] = list of tuples for cases:  [('Nom', 'Sg'), ('Acc', 'Sg')],
                ## 
            
            if base_analysis:
                # Modify results to show tva derivation
                for entry in base_analysis:
                    if isinstance(entry, list) and len(entry) >= 5:
                        entry[1] = f"{entry[1]} + tva"  # Mark as tva derivative
                        entry[4] = word  # Original form
                return base_analysis + ["tva"]
            
    return None

@lru_cache(maxsize=256)
def root_any_word(word, attempted_words=None, timed=False, session=None):
    

    if attempted_words is None:
        attempted_words = frozenset()
    
    result_roots = None

    # If the word has already been attempted, return None to avoid infinite loop
    if word in attempted_words:
        return None

    # Create a new frozenset with the current word added
    attempted_words = frozenset([word]).union(attempted_words)

    if timed:
        start_time = time.time()

    if word: 
        result_roots_name = SQLite_find_name(word, session=session)
    else:
        return None
    
    if timed:
        print(f"SQLite_find_name({word}) took {time.time() - start_time:.6f} seconds")

    if timed:
        start_time = time.time()
    result_roots_verb = SQLite_find_verb(word, session=session)
    if timed:
        print(f"SQLite_find_verb({word}) took {time.time() - start_time:.6f} seconds")

    if result_roots_name and result_roots_verb:
        result_roots = result_roots_name + result_roots_verb
    elif result_roots_name:
            result_roots = result_roots_name
    elif result_roots_verb:
        result_roots = result_roots_verb

    ### add abbreviation here
    if result_roots:
        for i in range(len(result_roots)):
            result = result_roots[i]
            # Get the second member of the list
            abbr = result[1]
            if abbr in type_map:
                result[1] = type_map[abbr]

        return result_roots

    # If no result is found, try replacements based on variableSandhi
    if word[-1] in variableSandhi:
        for replacement in variableSandhi[word[-1]]:
            tentative = word[:-1] + replacement
            if timed:
                start_time = time.time()
            if tentative not in attempted_words:
                print (f"tentative: {tentative}")
                print (f"attempted_words: {attempted_words}")
                attempt = root_any_word(tentative, attempted_words, timed, session=session)
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
        attempt = root_any_word(tentative, attempted_words, timed, session=session)
        if timed:
            print(f"root_any_word({tentative}) took {time.time() - start_time:.6f} seconds")
        if attempt is not None:
            return attempt
        
    ### possibility two, adding compound handling here 

    if result_roots is None:
        #print("to test with tva", word)
        tva_result = handle_tva(word, session=session)
        #print("tva_result", tva_result)
        if tva_result:
            return tva_result
    
    #print("to test with prefixes", word)
    
    for prefix in SANSKRIT_PREFIXES:
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            attempt = root_any_word(remainder, session=session)
            if attempt is not None:
                print("attempt", attempt)
                print("prefix", prefix)
                if prefix == 'ud': 
                    result = root_any_word('ut', session=session) + attempt
                else: 
                    result = root_any_word(prefix, session=session) + attempt
                for match in result: 
                    if len(match) == 5:
                        match[4] = word
                return result
            else: 
                for nested_prefix in SANSKRIT_PREFIXES:
                    if remainder.startswith(nested_prefix):
                        nested_remainder = remainder[len(nested_prefix):]
                        nested_attempt = root_any_word(nested_remainder, session=session)
                        if nested_attempt is not None:
                            result =  root_any_word(prefix, session=session) + root_any_word(nested_prefix, session=session) + nested_attempt
                            for match in result: 
                                if len(match) == 5:
                                    match[4] = word
                            print("working", result)
                            return result
            
    return None

