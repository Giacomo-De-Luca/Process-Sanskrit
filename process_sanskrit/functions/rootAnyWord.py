import time 
from process_sanskrit.functions.SQLiteFind import SQLite_find_name, SQLite_find_verb
from process_sanskrit.utils.loadResources import type_map
from process_sanskrit.utils.lexicalResources import variableSandhi
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES, samMap
from dataclasses import dataclass
from typing import Optional, List
 

##given a name finds the root


def root_any_word(word, attempted_words=None, timed=False, session=None):

    if word == 'api' or word == 'āpi':
        return ['api' , 'api' , ['api']]  
    
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


    result_roots_verb = SQLite_find_verb(word, session=session)

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
                #print (f"tentative: {tentative}")
                #print (f"attempted_words: {attempted_words}")
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

    #if result_roots is None:
        #print("to test with tva", word)
    #    tva_result = handle_tva(word, session=session)
        #print("tva_result", tva_result)
    #    if tva_result:
    #        return tva_result
    
    #print("to test with prefixes", word)
    
    for prefix in SANSKRIT_PREFIXES:
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            attempt = root_any_word(remainder, session=session)
            if attempt is not None:
                if prefix == 'ud': 
                    result = root_any_word('ut', session=session) + attempt
                else: 
                    prefix_root = root_any_word(prefix, session=session)
                    result = prefix_root + attempt if prefix_root else attempt
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
                            return result
            
    return None

