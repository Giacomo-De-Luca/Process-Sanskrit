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


import logging
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
from process_sanskrit.utils.transliterationUtils import transliterate

### import the sandhiSplitScorer and construct the scorer object. 

from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.dictionaryLookup import dict_search, multidict
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.hybridSplitter import hybrid_sandhi_splitter
from process_sanskrit.functions.inflect import inflect
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.databaseSetup import session_scope, with_session, requires_database



### get the version of the library

logging.basicConfig(level=logging.CRITICAL)


def preprocess(text, max_length=100, debug=False):

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

    if "o'" in text:
        text = re.sub(r"o'", "aḥ a", text)

    if text[0] == "'":
        text = 'a' + text[1:]

    text = regex.sub('[^\p{L}\'_%*-+]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def handle_special_characters(text: str, dict_names: Optional[Tuple[str, ...]] = None, session=None) -> Optional[List]:
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
    
    Returns:
        List containing processed entries if special handling occurred,
        None if no special handling was needed
    
    Examples:
        >>> handle_special_characters("deva*")  # Wildcard search
        >>> handle_special_characters("dev_")   # Pattern matching
        >>> handle_special_characters("deva-datta")  # Compound splitting
    """    
    # Handle wildcard search with asterisk
    if text.endswith('*'):
        transliterated_text = transliterate(text[:-1], "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        if not isinstance(voc_entry[0][2], list):
            return voc_entry
        return process(text[:-1])

    # Handle explicit wildcard search with _ or %
    if '_' in text or '%' in text:
        transliterated_text = transliterate(text, "IAST")
        voc_entry = dict_search([transliterated_text], *dict_names, session=session)
        if not isinstance(voc_entry[0][2], list):
            return voc_entry
        return process(text)

    # Handle pre-split compounds with - or + 
    if "-" in text or "+" in text:
        word_list = re.split(r'[-+]', text)
        processed_results = []
        for word in word_list:
            result = process(word)
            processed_results.extend(result)
        return processed_results

    return None  # Return None if no special cases matched


### roots should be replaced by output="roots" in the function signature
### by default, output = "detailed"
@requires_database
@with_session
def process(text, *dict_names, max_length=100, debug=False, mode="detailed", session=None):

    
    text = preprocess(text, max_length=max_length, debug=debug)

    ## if text is none return empty list
    if not text:
        if mode == "roots":
            return ""
        else:
            return []

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        check_special_characters = handle_special_characters(text, dict_names, session=session)
        if check_special_characters is not None:
            return check_special_characters

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        
        if text and text[-1] in sanskritFixedSandhiMap:
            text = text[:-1] + sanskritFixedSandhiMap[text[-1]]

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text, session=session)
        if debug == True:
            print("rooting result", result)

        if result is None and "ṅ" in text:
            ## this is removed, it was not triggering, and it was not clear if it was useful: or "ñ" in text
            tentative = text.replace("ṅ", "ṃ")
            attempt = root_any_word(tentative, session=session)
            if attempt is not None:
                result = attempt
        
        if result is None and "ṁ" in text:
            tentative = text.replace("ṁ", "ṃ")
            attempt = root_any_word(tentative, session=session)
            if attempt is not None:
                result = attempt

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "ch":
            #print("tentative", text)
            tentative = 'ś' + text[1:] 
            attempt = root_any_word(tentative, session=session)
            #print("attempt", attempt)
            if attempt is not None:
                result = attempt

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
            #if isinstance(result_vocabulary, list):
                
            #    if len(result[0]) > 4 and result[0][0] != result[0][4] and result[0][4] in DICTIONARY_REFERENCES.keys():
            #        replacement = dict_search([result[0][4]], *dict_names, session=session)
            #        if debug:
            #            print("replacement", replacement[0])
            #            print("len replacement", len(replacement[0]))
            #        if replacement is not None:
            #            result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary, debug=debug, mode=mode)
        else:
            ## if result is None, we try to find the word in the dictionary for exact match
            result_vocabulary = dict_search([text], *dict_names, session=session)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary, debug=debug, mode=mode)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case


    splitted_text = hybrid_sandhi_splitter(text)
    if debug == True:
        print("splitted_text_here", splitted_text)
    inflections = inflect(splitted_text, session=session) 
    if debug == True:
        print("inflections after splitting", inflections)
    inflections_vocabulary = dict_search(inflections, *dict_names, session=session)

    ## should this really be kept? 
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]
      
    return clean_results(inflections_vocabulary, debug=debug, mode=mode)


