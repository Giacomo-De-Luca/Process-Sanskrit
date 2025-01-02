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
### - call process with root_only=True to get only the root of all the words in a Sanskrit text. 


### packages and local modules import 


import logging
import re
import regex
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Dict, Tuple, Union, Optional
import time


from utils.lexicalResources import (
    variableSandhi, 
    sanskritFixedSandhiMap, 

    SANSKRIT_PREFIXES
)
from utils.transliterationUtils import (

    anythingToIAST,

)
from utils.loadResources import mwdictionaryKeys
from utils.databaseSetup import Session, engine, Base

session = Session()

### import the sandhiSplitScorer and construct the scorer object. 

from functions.rootAnyWord import root_any_word
from functions.dictionaryLookup import get_voc_entry, multidict
from functions.cleanResults import clean_results
from functions.hybridSplitter import hybrid_sandhi_splitter
from functions.inflect import inflect





### get the version of the library

__version__ = "0.3"
def print_version():
    print(f"Version: {__version__}")

logging.basicConfig(level=logging.CRITICAL)



def process(text, *dict_names, max_length=100, debug=False, root_only=False, ):


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

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        ## if the text end with a *, remove it and try to find the word in the dictionary for exact match
        if text[-1] == '*':
            voc_entry = get_voc_entry([anythingToIAST(text[:-1])], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text[:-1])

        ## if in the text there is a _, use wildcard search directly

        if '_' in text or '%' in text:
            print("wildcard search")
            voc_entry = get_voc_entry([anythingToIAST(text)], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text)

        text = anythingToIAST(text)

        ## if the text is already sandhi split keep it sandhi split
        if "-" in text or "+" in text:
            # Use re.split to split by either "-" or "+"
            word_list = re.split(r'[-+]', text)
            processed_results = []
            for word in word_list:
                result = process(word)
                processed_results = processed_results + result
            return processed_results
        

        ## remove all non-alphabetic characters
        text = regex.sub('[^\p{L}\']', '', text)

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        if text[0] == "'":
            text = 'a' + text[1:]
        
        if text[-1] in sanskritFixedSandhiMap:
            text = text[:-1] + sanskritFixedSandhiMap[text[-1]]

        elif text[-1] == 'ś':
            text = text[:-1] + 'ḥ'

        #print("text", text)
        if "o'" in text:
            modified_text = re.sub(r"o'", "aḥ a", text)
            #print("modified_text", modified_text)
            result = process(modified_text)
            return result

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text)


        ## c'è una funzione che fa esattamente questo dopo... probabilmente si può evitare. 
        # Look for prefixes only at the start
        if result is None:
            print("text with prefixes", text)
            for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                if text.startswith(prefix):
                    remainder = text[len(prefix):]
                    print("remainder", remainder)
                    attempt = root_any_word(remainder)
                    if attempt is not None:
                        result = [prefix] + attempt
                        break
            ## commented out as it breaks a lot of words like samadhi // it matches too much
            #       else:

                        ### breaks with ps.process("samādhipariṇāmaḥ")
                        #for nested_prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                            #if remainder.startswith(nested_prefix):
                                #nested_remainder = remainder[len(prefix):]
                                #print("nested_remainder", nested_remainder)
                                #nested_attempt = root_any_word(nested_remainder)
                                #if nested_attempt is not None:
                                    #result = [prefix] + [nested_prefix] + nested_attempt
                                    #break

        ## if the word ends with a M, try to correct it for different conventions  and find the word in the dictionary for exact match
        # Check if the word ends with a character that is a key in variableSandhiSLP1
        if result is None and text[-1] in variableSandhi:
            for replacement in variableSandhi[text[-1]]:
                tentative = text[:-1] + replacement
                attempt = root_any_word(tentative)
                if attempt is not None:
                    result = attempt
                    break

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
                
                if len(result[0]) > 4 and result[0][0] != result[0][4] and result[0][4] in mwdictionaryKeys:
                    replacement = get_voc_entry([result[0][4]], *dict_names)
                    print("replacement", replacement[0])
                    print("len replacement", len(replacement[0]))
                    if replacement is not None:
                        result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary, debug=debug, root_only=root_only)
        else:
            query = [text]
            #print("query", query)
            result_vocabulary = get_voc_entry(query, *dict_names)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary, debug=debug, root_only=root_only)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case
    if debug == True:
        print("pre_splitted_text", text)
    #print("transliterate to split", text)
    splitted_text = hybrid_sandhi_splitter(text)
    if debug == True:
        print("splitted_text", splitted_text)
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary, debug=debug, root_only=root_only)




def processTimed(text, *dict_names, max_length=100, debug=True, root_only=False, recordTime=True ):

    start_process = time.time() if recordTime else None
    times = {} if recordTime else None

    start_time = time.time() if recordTime else None


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

    if recordTime:
        times['text_preprocessing'] = time.time() - start_time


    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        start_time = time.time() if recordTime else None

        ## if the text end with a *, remove it and try to find the word in the dictionary for exact match
        if text[-1] == '*':
            voc_entry = get_voc_entry([anythingToIAST(text[:-1])], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text[:-1])

        ## if in the text there is a _, use wildcard search directly

        if '_' in text or '%' in text:
            print("wildcard search")
            voc_entry = get_voc_entry([anythingToIAST(text)], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text)

        
        text = anythingToIAST(text)

        if recordTime:
            times['transliteration'] = time.time() - start_time

        start_time = time.time() if recordTime else None

        ## if the text is already sandhi split keep it sandhi split
        if "-" in text or "+" in text:
            # Use re.split to split by either "-" or "+"
            word_list = re.split(r'[-+]', text)
            processed_results = []
            for word in word_list:
                result = process(word)
                processed_results = processed_results + result
            return processed_results
        

        ## remove all non-alphabetic characters
        text = regex.sub('[^\p{L}\']', '', text)

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        if text[0] == "'":
            text = 'a' + text[1:]
        
        if text[-1] in sanskritFixedSandhiMap:
            text = text[:-1] + sanskritFixedSandhiMap[text[-1]]

        elif text[-1] == 'ś':
            text = text[:-1] + 'ḥ'

        #print("text", text)
        if "o'" in text:
            modified_text = re.sub(r"o'", "aḥ a", text)
            #print("modified_text", modified_text)
            result = process(modified_text)
            return result

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text)

        if recordTime:
            times['root_analysis'] = time.time() - start_time

        start_time = time.time() if recordTime else None


        ## c'è una funzione che fa esattamente questo dopo... probabilmente si può evitare. 
        # Look for prefixes only at the start
        if result is None:
            print("text with prefixes", text)
            for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                if text.startswith(prefix):
                    remainder = text[len(prefix):]
                    print("remainder", remainder)
                    attempt = root_any_word(remainder)
                    if attempt is not None:
                        result = [prefix] + attempt
                        break
            ## commented out as it breaks a lot of words like samadhi // it matches too much
            #       else:

                        ### breaks with ps.process("samādhipariṇāmaḥ")
                        #for nested_prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                            #if remainder.startswith(nested_prefix):
                                #nested_remainder = remainder[len(prefix):]
                                #print("nested_remainder", nested_remainder)
                                #nested_attempt = root_any_word(nested_remainder)
                                #if nested_attempt is not None:
                                    #result = [prefix] + [nested_prefix] + nested_attempt
                                    #break

        ## if the word ends with a M, try to correct it for different conventions  and find the word in the dictionary for exact match
        # Check if the word ends with a character that is a key in variableSandhiSLP1
        if result is None and text[-1] in variableSandhi:
            for replacement in variableSandhi[text[-1]]:
                tentative = text[:-1] + replacement
                attempt = root_any_word(tentative)
                if attempt is not None:
                    result = attempt
                    break

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
                
                if len(result[0]) > 4 and result[0][0] != result[0][4] and result[0][4] in mwdictionaryKeys:
                    replacement = get_voc_entry([result[0][4]], *dict_names)
                    print("replacement", replacement[0])
                    print("len replacement", len(replacement[0]))
                    if replacement is not None:
                        result_vocabulary.insert(0, replacement[0])
            if recordTime:
                times['dictionary_lookup'] = time.time() - start_time

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary, debug=debug, root_only=root_only)
        else:
            query = [text]
            #print("query", query)
            result_vocabulary = get_voc_entry(query, *dict_names)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary, debug=debug, root_only=root_only)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case
    if debug == True:
        print("pre_splitted_text", text)
    #print("transliterate to split", text)

    start_time = time.time() if recordTime else None
    splitted_text = hybrid_sandhi_splitter(text)
    if recordTime:
        times['sandhi_splitting'] = time.time() - start_time

    start_time = time.time() if recordTime else None
    if debug == True:
        print("splitted_text", splitted_text)
    inflections = inflect(splitted_text) 
    if recordTime:
        times['inflection_analysis'] = time.time() - start_time

    start_time = time.time() if recordTime else None
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]      
    if recordTime:
        times['vocabulary_lookup'] = time.time() - start_time
        times['total_time'] = time.time() - start_process 

    # Add timing data to the result if requested
    result = clean_results(inflections_vocabulary, debug=debug, root_only=root_only)
    if recordTime:
        if debug:
            print("\n=== Processing Time Summary ===")
            print(f"Total execution time: {times['total_time']:.4f} seconds")
            print("\nBreakdown by operation:")
            for operation, duration in times.items():
                if operation != 'total_time':
                    percentage = (duration / times['total_time']) * 100
                    print(f"{operation.replace('_', ' ').title()}: {duration:.4f}s ({percentage:.1f}%)")
        return result, times

    return result



