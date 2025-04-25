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
### - call process with roots=True to get only the root of all the words in a Sanskrit text. 


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
from process_sanskrit.utils.transliterationUtils import (
    anythingToIAST,
)
from process_sanskrit.utils.databaseSetup import Session, engine, Base

### import the sandhiSplitScorer and construct the scorer object. 

from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.dictionaryLookup import get_voc_entry, multidict
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.hybridSplitter import hybrid_sandhi_splitter
from process_sanskrit.functions.inflect import inflect
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES

session = Session()



### get the version of the library

__version__ = "0.3"
def print_version():
    print(f"Version: {__version__}")

logging.basicConfig(level=logging.CRITICAL)


def preprocess(text, max_length=100, debug=False):

    text = anythingToIAST(text)

    ## if the text is too long, we try to trim it to the last whitespace
    if len(text) > max_length:
        last_space_index = text[:max_length].rfind(' ')
        if last_space_index == -1:
            text = text[:max_length]
        else:
            # Trim up to the last whitespace
            text = text[:last_space_index]

    ## jj occours only in sandhi and we know it should be split into two words. 
    if 'jj' in text:
        text = text.replace('jj', 'j j')

    if "o'" in text:
        text = re.sub(r"o'", "aḥ a", text)

    if text[0] == "'":
        text = 'a' + text[1:]

    return text


def handle_special_characters(text: str, dict_names: Optional[Tuple[str, ...]] = None) -> Optional[List]:
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
        transliterated_text = anythingToIAST(text[:-1])
        voc_entry = get_voc_entry([transliterated_text], *dict_names)
        if voc_entry is not None:
            return voc_entry
        return process(text[:-1])

    # Handle explicit wildcard search with _ or %
    if '_' in text or '%' in text:
        transliterated_text = anythingToIAST(text)
        voc_entry = get_voc_entry([transliterated_text], *dict_names)
        if voc_entry is not None:
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



def process(text, *dict_names, max_length=100, debug=False, roots="none", count_types = False):


    counts = {"word_calls": 1, "hybrid_splitter": 0, "compound_calls": 0} if count_types else None
    


    text = preprocess(text, max_length=max_length, debug=debug)

    ## remove all non-alphabetic characters
    text = regex.sub('[^\p{L}\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    ## aggiungi alla versione online
    text = text.strip()

    ## if text is none return empty list
    if not text:
        if roots == "roots":
            return ""
        else:
            return []

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        check_special_characters = handle_special_characters(text, dict_names)
        if check_special_characters is not None and not count_types:
            return check_special_characters


        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        

        if text and text[-1] in sanskritFixedSandhiMap:
            text = text[:-1] + sanskritFixedSandhiMap[text[-1]]

         ## move this to the fixed sandhi map
        elif text[-1] == 'ś':
            text = text[:-1] + 'ḥ'

        #print("text", text)
        #if "o'" in text:
        #    modified_text = re.sub(r"o'", "aḥ a", text)
            #print("modified_text", modified_text)
        #    result = process(modified_text)
        #    return result

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text)

        if result is None and "ṅ" in text or "ñ" in text:
            tentative = text.replace("ṅ", "ṃ")
            attempt = root_any_word(tentative)
            if attempt is not None:
                result = attempt


        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "ch":
            #print("tentative", text)
            tentative = 'ś' + text[1:] 
            attempt = root_any_word(tentative)
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
            result_vocabulary = get_voc_entry(result, *dict_names)

            if debug == True: 
                print("result_vocabulary", result_vocabulary)

            ## if the word is inside the dictionary, we return the entry directly, since it will be accurate. 
            if isinstance(result_vocabulary, list):
                
                if len(result[0]) > 4 and result[0][0] != result[0][4] and result[0][4] in DICTIONARY_REFERENCES.keys():
                    replacement = get_voc_entry([result[0][4]], *dict_names)
                    if debug:
                        print("replacement", replacement[0])
                        print("len replacement", len(replacement[0]))
                    if replacement is not None:
                        result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            if count_types:
                return counts
            return clean_results(result_vocabulary, debug=debug, roots=roots)
        else:
            ## if result is None, we try to find the word in the dictionary for exact match
            result_vocabulary = get_voc_entry([text], *dict_names)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                if count_types:
                    return counts
                return clean_results(result_vocabulary, debug=debug, roots=roots)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case

    if count_types:
        counts["hybrid_splitter"] += 1
        compound_calls = hybrid_sandhi_splitter(text, count_types=True)
        counts["compound_calls"] += compound_calls
        return counts
    else:
        splitted_text = hybrid_sandhi_splitter(text, detailed_output=debug)

    if debug == True:
        print("splitted_text", splitted_text)
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]
      
    return clean_results(inflections_vocabulary, debug=debug, roots=roots)




def process_texts(texts, errors=False):
    """
    Process multiple Sanskrit texts and count function calls.
    
    Parameters:
    - texts: A string or list of strings to process
    - errors: If True, return error words list along with counts
    
    Returns:
    - Dictionary with counts for each function type
    - List of words that caused errors (if errors=True)
    """
    # Initialize counts
    total_counts = {"word_calls": 0, "hybrid_splitter": 0, "compound_calls": 0}
    error_words = []  # List to track words that caused errors
    
    # Convert single string to list and handle lists
    if isinstance(texts, str):
        text_list = texts.split()
    else:
        # If texts is already a list, flatten it into words
        text_list = []
        for item in texts:
            if item:
                text_list.extend(item.split())
    
    # Process each word that contains alphabetical characters
    for word in text_list:
        # Check if the word contains at least one alphabetical character
        if regex.search(r'\p{L}', word):
            try:
                counts = process(
                    word, 
                    count_types=True
                )
                
                # Aggregate counts
                total_counts["word_calls"] += counts["word_calls"]
                total_counts["hybrid_splitter"] += counts["hybrid_splitter"]
                total_counts["compound_calls"] += counts["compound_calls"]
            except Exception as e:
                # Log the error and continue with the next word
                error_words.append(word)
                print(f"Error processing word '{word}': {str(e)}")
                continue
    
    # Add error count to the result
    total_counts["errors"] = len(error_words)

    if errors:
        return total_counts, error_words
    
    return total_counts