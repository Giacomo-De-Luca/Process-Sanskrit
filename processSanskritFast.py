### packages and local modules import 

import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import ast
from utils.detectTransliteration import detect
import logging
import json
import sqlite3
import re
import pandas as pd
from sanskrit_parser import Parser
from tabulate import tabulate
import regex
from itertools import groupby
import unicodedata
from typing import List, Dict, Any
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
import os

### import local resources from the utils module

### import lexical resources from the utils.lexicalResources module

from utils.lexicalResources import (
    variableSandhiSLP1, 
    sanskritFixedSandhiMapSLP1, 
    VOWEL_SANDHI_INITIALS, 
    SANDHI_VARIATIONS, 
    SANDHI_VARIATIONS_IAST, 
    SANSKRIT_PREFIXES
)

### import transliteration functions from the utils.transliteration_utils module
### the anythingTo function first detects the input scheme and then transliterates it to the output scheme

from utils.transliterationUtils import (
    transliterateSLP1IAST,
    transliterateIASTSLP1,
    transliterateSLP1HK,
    transliterateDEVSLP1,
    anythingToSLP1,
    anythingToIAST,
    anythingToHK,
    transliterateAnything
)

### import the sandhiSplitScorer and construct the scorer object. 
    
from functions.sandhiSplitScorer import SandhiSplitScorer

from functions.enhancedSandhiSplitter import enhanced_sandhi_splitter

scorer = SandhiSplitScorer()

with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

with open('resources/MWKeysOnlySLP1.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeysSLP1 = json.load(f)




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


### get the version of the library

__version__ = "0.3"
def print_version():
    print(f"Version: {__version__}")

logging.basicConfig(level=logging.CRITICAL)



#from dotenv import load_dotenv

## If heroku postgres
#load_dotenv()
#DATABASE_URL = os.environ['DATABASE_URL']

#if local postgres

#DATABASE_URL = "postgresql+psycopg2://postgres:again@localhost:5432/sanskritmagicdb"
#DATABASE_URL = os.getenv("DATABASE_URL")
#DATABASE_URL = os.getenv("postgres://u5o7c326q19pvp:pdb08c4f74fd1c63df61a559fc3d7a261ac3c65a24df6cfd06fbb2ad511143f0d@c3cj4hehegopde.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d61ijmaljbh829")

#if DATABASE_URL.startswith("postgres://"):
    #DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

#if using SQLite



from utils.database_config import db


session = db.get_session()
SplitCache = db.SplitCache



##given a name finds the root

def SQLite_find_name(name):

    outcome = []    

    def query1(word):

        session = Session()
        try:
            query_builder = text("SELECT * FROM lgtab2 WHERE key = :word")
            results = session.execute(query_builder, {'word': word}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []
        finally:
            session.close()
        return results

    results = query1(name)
    
    if not results:  # If query1 didn't find any results
        if name[-1] == 'M':
            name = name[:-1] + 'm'
            results = query1(name)
    
    for inflected_form, type, root_form in results: 
        if not root_form:  # If root_form is None or empty
            return  # End the function

        def query2(root_form: str, type: str):

            session = Session()
            try:
                query_builder2 = text("SELECT * FROM lgtab1 WHERE stem = :root_form and model = :type ")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []
            finally:
                session.close()
            return results
        
        result = query2(root_form, type)
        word_refs = re.findall(r",([a-zA-Z]+)",result[0][2])[0]
        inflection_tuple = result[0][3]  # Get the first element of the first tuple
        inflection_words = inflection_tuple.split(':') 

        ##transliterate back the result to IAST for readability
        inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
        query_transliterateIAST = transliterateSLP1IAST(name)

        ##make Inflection Table
        indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]
        rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
        coltitles = ["Sg", "Du", "Pl"]

        if indices:
            row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
        else: 
            row_col_names = None
        outcome.append([word_refs, type, row_col_names, inflection_wordsIAST, transliterateSLP1IAST(name)])

    return outcome



def SQLite_find_verb(verb):
    
    root_form = None

    def query1(verb):

        session = Session()
        try:
            query_builder = text("SELECT * FROM vlgtab2 WHERE key = :verb")
            results = session.execute(query_builder, {'verb': verb}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []
        finally:
            session.close()
        return results

    result = query1(verb)
    
    for inflected_form, type, root_form in result:

        if not root_form:  # If root_form is None or empty
            return  # End the function
        type_var = type

        def query2(root_form: str, type: str):

            session = Session()
            try:
                query_builder2 = text("SELECT * FROM vlgtab1 WHERE stem = :root_form and model = :type")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []
            finally:
                session.close()
            return results
        
        result = query2(root_form, type)
    
    selected_tuple = None

    # Iterate over the result list
    for model, stem, refs, data in result:
        if model == type_var:  # If the model matches type_var
            ref_word = re.search(",([a-zA-Z]+)", refs).group(1)
            if stem != ref_word:
                stem= ref_word
                #print("ref_word, stem", ref_word, stem)
                selected_tuple = (model, stem, refs, data)  # Get the entire tuple
                break  # Exit the loop
            selected_tuple = (model, stem, refs, data)  # Get the entire tuple
            break  # Exit the loop

    if selected_tuple is None:
        #print("No matching model found in result")
        return
    

    # Now you can use selected_tuple
    inflection_tuple = selected_tuple[3]  # Get the 'data' element of the tuple
    inflection_words = inflection_tuple.split(':') 
    
    ##transliterate back the result to IAST for readability
    
    inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
    query_transliterateIAST = transliterateSLP1IAST(verb)
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]

    # Define row and column titles
    rowtitles = ["First", "Second", "Third"]
    coltitles = ["Sg", "Du", "Pl"]


    if indices:
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
    else:
        row_col_names = None
        
    return [[stem, type_var, row_col_names, inflection_wordsIAST, transliterateSLP1IAST(verb)]]


## also map to the type.
# Read the Excel file into a DataFrame
type_map = pd.read_excel('resources/type_map.xlsx')

def root_any_word(word, attempted_words=None):
    if attempted_words is None:
        attempted_words = set()

    # If the word has already been attempted, return None to avoid infinite loop
    if word in attempted_words:
        return None

    # Add the current word to the set of attempted words
    attempted_words.add(word)

    result_roots_name = SQLite_find_name(word)  
    result_roots_verb = SQLite_find_verb(word) 

    if result_roots_name:
        if result_roots_verb:
            result_roots = result_roots_name + result_roots_verb
        else:
            result_roots = result_roots_name
    else:
        result_roots = result_roots_verb

    if result_roots:
        for i in range(len(result_roots)):
            result = result_roots[i]
            # Get the second member of the list
            abbr = result[1]
            # Find the matching value in the 'abbr' column
            match = type_map[type_map['abbr'] == abbr]
            
            if not match.empty:
                description = match['description'].values[0]
                result[1] = description
                result_roots[i] = result
        return result_roots

    # If no result is found, try replacements based on variableSandhiSLP1
    if word[-1] in variableSandhiSLP1:
        for replacement in variableSandhiSLP1[word[-1]]:
            tentative = word[:-1] + replacement
            attempt = root_any_word(tentative, attempted_words)
            if attempt is not None:
                return attempt
    
    samMap = {
            'sam' : 'saM',
            'saM' : 'sam',
            'saN' : 'saM',
            'san' : 'saM',
            'saY' : 'saM',
        }

        ##different spellings for sam, - it is so common that it deserves its own rule
    if word[0:3] in samMap:
            print("tentative", word)
            tentative = samMap[word[0:3]] + word[3:]
            attempt = root_any_word(tentative, attempted_words)
            print("attempt", attempt)
            if attempt is not None:
                result = attempt
                return attempt

    return None





        ##if the dictionary approach fails, try the iterative approach:
import time

def inflect(splitted_text):
    roots = []
    prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'aBi', 'ni', 'pari', 'prati', 'parA', 'ava', 'aDi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duH']  # Add more prefixes as needed
    i = 0
    while i < len(splitted_text):
        word = splitted_text[i]
        print(f"Processing word: {word}")
        if word in prefixes and i + 1 < len(splitted_text):
            next_word = splitted_text[i + 1]
            print(f"Found prefix: {word}, next word: {next_word}")
            if word == 'sam':
                combined_words = ['sam' + next_word, 'saM' + next_word]
            elif word == 'vi':
                combined_words = ['vi' + next_word, 'vy' + next_word]
            else:
                combined_words = [word + next_word]

            rooted = None
            for combined_word in combined_words:
                start_time = time.time()
                rooted = root_any_word(combined_word)
                print(f"root_any_word({combined_word}) took {time.time() - start_time:.6f} seconds")
                if rooted is not None:
                    break  # Exit loop if a valid root is found

            if rooted is not None:
                roots.extend(rooted)
                i += 2  # Skip next word since it's part of the combined word
                continue
            else:
                start_time = time.time()
                rooted_word = root_any_word(word)
                print(f"root_any_word({word}) took {time.time() - start_time:.6f} seconds")
                if rooted_word is not None:
                    roots.extend(rooted_word)
                else:
                    start_time = time.time()
                    compound_try = root_compounds(word)
                    print(f"root_compounds({word}) took {time.time() - start_time:.6f} seconds")
                    if compound_try is not None:
                        roots.extend(compound_try)
                    else:
                        roots.append(word)
                i += 1  # Move to next word
        else:
            start_time = time.time()
            rooted = root_any_word(word)
            print(f"root_any_word({word}) took {time.time() - start_time:.6f} seconds")
            if rooted is not None:
                roots.extend(rooted)
            else:
                start_time = time.time()
                compound_try = root_compounds(word)
                print(f"root_compounds({word}) took {time.time() - start_time:.6f} seconds")
                if compound_try is not None:
                    roots.extend(compound_try)
                else:
                    roots.append(word)
            i += 1

    # Transliterate roots
    for j in range(len(roots)):
        if isinstance(roots[j], list):
            start_time = time.time()
            roots[j][0] = transliterateSLP1IAST(roots[j][0].replace('-', ''))
            print(f"transliterateSLP1IAST({roots[j][0]}) took {time.time() - start_time:.6f} seconds")
        else:
            start_time = time.time()
            roots[j] = transliterateSLP1IAST(roots[j].replace('-', ''))
            print(f"transliterateSLP1IAST({roots[j]}) took {time.time() - start_time:.6f} seconds")
    return roots


## bug with process("nīlotpalapatrāyatākṣī")    



def try_match_with_prefixes(word, debug=False):
    """
    Attempts to match a word by checking for prefixes ONLY at the start.
    Does not recursively look for prefixes in the remainder.
    
    Args:
        word: The word to analyze in SLP1 notation
        debug: Whether to print debug information
    
    Returns:
        tuple: (matched_word, end_letter) or None if no match found
    """
    if debug:
        print(f"\nTrying to match word with prefixes: {word}")
    
    # First try the whole word as-is
    if word in mwdictionaryKeysSLP1:
        if debug:
            print(f"Found direct match in dictionary: {word}")
        return (word, word[-1])
    
    # Look for prefixes only at the start
    for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            if debug:
                print(f"Found prefix {prefix}, trying remainder: {remainder}")
            
            # For the remainder, just try a direct dictionary match
            if remainder in mwdictionaryKeysSLP1:
                if debug:
                    print(f"Found remainder in dictionary: {remainder}")
                return (word[:len(prefix) + len(remainder)], word[len(prefix) + len(remainder) - 1])    
    
    if debug:
        print("No match found after prefix analysis")
    return None

def dict_word_iterative(word, debug=False):
    """
    Dictionary lookup that integrates prefix matching with sandhi variations.
    For each possible word length, tries:
    1. Direct dictionary match
    2. Prefix matches
    3. Sandhi variations (with prefix checks on each variation)
    """
    temp_word = word
    best_match = None
    best_length = 0
    
    if debug:
        print(f"Attempting to match word: {word}")
    
    # First try root_any_word on complete word
    root_result = root_any_word(temp_word)
    if root_result:
        if debug:
            print(f"Found inflected form: {temp_word}")
        return (temp_word, temp_word[-1])
    
    while temp_word and len(temp_word) > 1:
        # Try direct dictionary match
        if temp_word in mwdictionaryKeysSLP1:
            if len(temp_word) > best_length:
                if debug:
                    print(f"Found dictionary match: {temp_word}")
                best_match = temp_word
                best_length = len(temp_word)
        
        # Try prefix matches on current word
        prefix_match = try_match_with_prefixes(temp_word, debug)
        if prefix_match and len(prefix_match[0]) > best_length:
            if debug:
                print(f"Found prefix match: {prefix_match[0]}")
            best_match = prefix_match[0]
            best_length = len(prefix_match[0])
        
        # Try sandhi variations and check prefixes on each variant
        last_char = temp_word[-1]
        if last_char in SANDHI_VARIATIONS:
            for variant in SANDHI_VARIATIONS[last_char]:
                test_word = temp_word[:-1] + variant
                
                # Try direct match of sandhi variant
                if test_word in mwdictionaryKeysSLP1:
                    if len(test_word) > best_length:
                        if debug:
                            print(f"Found match with sandhi variation: {test_word}")
                        best_match = test_word
                        best_length = len(test_word)
                
                # Try prefix match on sandhi variant
                prefix_match = try_match_with_prefixes(test_word, debug)
                if prefix_match and len(prefix_match[0]) > best_length:
                    if debug:
                        print(f"Found prefix match on sandhi variant: {prefix_match[0]}")
                    best_match = prefix_match[0]
                    best_length = len(prefix_match[0])
        
        # If we found a match of full length, stop here
        if best_match and len(best_match) == len(temp_word):
            break
            
        temp_word = temp_word[:-1]
    
    if best_match:
        return (best_match, word[len(best_match)-1])
    
    if debug:
        print("No match found")
    return None



def root_compounds(word, debug=False, inflection=True, ):
    """
    Analyzes a long Sanskrit compound with improved sandhi handling between segments.
    """
    if debug:
        print("\nStarting analysis of:", word)
        print("Length:", len(word))

        # Handle initial apostrophe (avagraha)
    if word.startswith("'"):
        word = 'a' + word[1:]
        
    roots = []
    current_pos = 0
    
    while current_pos < len(word):
        remaining = word[current_pos:]
        if debug:
            print(f"\nAnalyzing segment starting at position {current_pos}: {remaining}")
            
        # Try the base case first
        best_match = None
        best_length = 0
        first_match = dict_word_iterative(remaining)

        
        if first_match:
            best_match = first_match
            best_length = len(first_match[0])
            if debug:
                print(f"Found base match: {first_match[0]} (ends with {first_match[1]})")
        
        # If the previous match ended in a vowel that can cause sandhi,
        # try analyzing the remaining text with added initial vowels
        if current_pos > 0:
            prev_end = word[current_pos - 1]
            if prev_end in VOWEL_SANDHI_INITIALS:
                if debug:
                    print(f"Trying sandhi variations for previous ending {prev_end}")
                
                for initial_vowel in VOWEL_SANDHI_INITIALS[prev_end]:
                    test_word = initial_vowel + remaining
                    if debug:
                        print(f"Trying with added {initial_vowel}: {test_word}")

                    # First try root_any_word with the sandhi-modified version
                    root_result = root_any_word(test_word)
                    if root_result:
                        if debug:
                            print(f"Found inflected form with sandhi: {test_word}")
                        test_match = (test_word, test_word[-1])
                        if len(test_word) > best_length:
                            best_match = test_match
                            best_length = len(test_word)
                            continue
                    
                    # If no inflected form found, try dictionary match
                    test_match = dict_word_iterative(test_word)
                    if test_match and len(test_match[0]) > best_length:
                        best_match = test_match
                        best_length = len(test_match[0])
                        if debug:
                            print(f"Found better match with sandhi: {test_match[0]}")


            if remaining.startswith('C'):
                test_word = 'S' + remaining[1:]
                if debug:
                    print(f"Trying with S instead of C: {test_word}")
                test_match = dict_word_iterative(test_word)
                if test_match and len(test_match[0]) > best_length:
                    best_match = test_match
                    best_length = len(test_match[0])
                    if debug:
                        print(f"Found better match with S: {test_match[0]}")
        
        if not best_match:
            if debug:
                print("No match found, moving forward one character")
            current_pos += 1
            continue
            
        matched_word, end_letter = best_match
        
        if inflection==True:
            # Process the matched word
            root_entry = root_any_word(matched_word)
            if root_entry:
                if isinstance(root_entry, list):
                    roots.extend(root_entry)
                else:
                    roots.append(root_entry)
            else:
                roots.append(matched_word)
        else:
            roots.append(matched_word)
        
        # Adjust position based on whether we used an added vowel
        vowel_adjustment = 1 if (current_pos > 0 and 
                                word[current_pos - 1] in VOWEL_SANDHI_INITIALS and 
                                matched_word[0] in VOWEL_SANDHI_INITIALS[word[current_pos - 1]]) else 0
        current_pos += max(len(matched_word) - vowel_adjustment, 1)

        if debug:
            print(f"Advanced position by {len(matched_word) - vowel_adjustment}")
    
    return roots

def process_root_result(root_result: Union[List, str]) -> str:
    """
    Process a single element from root_compounds output into a simple string.
    
    Parameters:
    - root_result: Either a string or a list containing morphological analysis
    
    Returns:
    - str: The base form of the word
    """
    if isinstance(root_result, str):
        return transliterateSLP1IAST(root_result)
    # If it's a list, the last element is usually the base form
    # Lists from root_compounds typically end with the base form
    return transliterateSLP1IAST(root_result[0])



def hybrid_sandhi_splitter(
    text_to_split: str,
    cached: bool = False,
    attempts: int = 10,
    detailed_output: bool = False,
    score_threshold: float = 0.535
) -> Union[List[str], Tuple[List[str], float, Dict, List]]:
    """
    Enhanced sandhi splitter that combines statistical splitting with root compound analysis.
    Processes complex root analysis output into scoreable word lists.
    
    Parameters:
    - text_to_split: Text to split
    - cached: Whether to use caching
    - attempts: Number of splitting attempts for statistical method
    - detailed_output: If True, returns additional scoring information
    - score_threshold: Minimum score to accept statistical split
    """
    # First try our enhanced statistical splitter
    if detailed_output:
        stat_split, stat_score, stat_subscores, all_splits = enhanced_sandhi_splitter(
            text_to_split, cached, attempts, detailed_output=True
        )
        if len(stat_split) == 1:
            stat_score = 0 
    else:
        stat_split = enhanced_sandhi_splitter(
            text_to_split, cached, attempts, detailed_output=False
        )
        # Get the score for comparison
        if len(stat_split) == 1:
            stat_score = 0 
        else: 
            stat_score, stat_subscores = scorer.score_split(text_to_split, stat_split)


    # If score is good enough, return statistical result
    if stat_score >= score_threshold:
        print("stat_score", stat_score)
        print("stat_split", stat_split)
        if detailed_output:
            return stat_split, stat_score, stat_subscores, all_splits
        return stat_split

    # If score is too low, try root compound analysis
    try:
        print("text_to_split", text_to_split)
        root_analysis = root_compounds(transliterateIASTSLP1(text_to_split), inflection=False)
        print("root_analysis", root_analysis)
        if root_analysis:
            # Process the root analysis results into a simple word list
            root_split = [process_root_result(item) for item in root_analysis]
            root_split = [x for i, x in enumerate(root_split) if i == 0 or x != root_split[i-1]]
            print("root_split", root_split)
            
            # Score the processed root split
            root_score, root_subscores = scorer.score_split(text_to_split, root_split)
            print("root_score", root_score)
            print("root_subscores", root_subscores)

            # Compare scores and choose the better result
            if root_score > stat_score:
                if detailed_output:
                    # Add root split to all_splits for reference
                    all_splits = [(root_split, root_score, root_subscores)] + (all_splits if all_splits else [])
                    return root_split, root_score, root_subscores, all_splits
                return root_split

    except Exception as e:
        print(f"Root compound analysis failed: {str(e)}")

    # Fall back to statistical split if root analysis fails or scores lower
    print("stat_split", stat_split)
    if detailed_output:
        return stat_split, stat_score, stat_subscores, all_splits
    return stat_split






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
            print("wildcard search", anythingToIAST)
            voc_entry = get_voc_entry([anythingToIAST(text)], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text)

        ## if the text is already sandhi split keep it sandhi split
        if "-" in text or "+" in text:
            # Use re.split to split by either "-" or "+"
            word_list = re.split(r'[-+]', text)
            processed_results = []
            for word in word_list:
                result = process(word)
                processed_results = processed_results + result
            return processed_results
        
        #print("single_word", text)
        transliterated_text = anythingToSLP1(text)     
        #print("transliterated_text", transliterated_text)

        ## remove all non-alphabetic characters
        text = regex.sub('[^\p{L}\']', '', transliterated_text)

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        if text[0] == "'":
            text = 'a' + text[1:]
        
        if text[-1] in sanskritFixedSandhiMapSLP1:
            text = text[:-1] + sanskritFixedSandhiMapSLP1[text[-1]]

        elif text[-1] == 'S':
            text = text[:-1] + 'H'

        #print("text", text)
        if "o'" in text:
            modified_text = re.sub(r"o'", "aH a", text)
            #print("modified_text", modified_text)
            result = process(modified_text)
            return result

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text)

        # Look for prefixes only at the start
        if result is None:
            for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                if text.startswith(prefix):
                    remainder = text[len(prefix):]
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
        if result is None and text[-1] in variableSandhiSLP1:
            for replacement in variableSandhiSLP1[text[-1]]:
                tentative = text[:-1] + replacement
                attempt = root_any_word(tentative)
                if attempt is not None:
                    result = attempt
                    break

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "C":
            #print("tentative", text)
            tentative = 'S' + text[1:] 
            attempt = root_any_word(tentative)
            #print("attempt", attempt)
            if attempt is not None:
                result = attempt
        
        if result is not None:
            if debug == True: 
                print("Getting some results with no splitting here:", result)

            for i, res in enumerate(result):
                if isinstance(res, str):
                    result[i] = transliterateSLP1IAST(res.replace('-', ''))
                elif isinstance(res, list):
                    if isinstance(res[0], str):
                        res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
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
            query = [transliterateSLP1IAST(text)]
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
    text = transliterateSLP1IAST(text)
    #print("transliterate to split", text)
    splitted_text = hybrid_sandhi_splitter(text)
    if debug == True:
        print("splitted_text", splitted_text)
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary, debug=debug, root_only=root_only)

import time

def processTimed(text, *dict_names, max_length=100, debug=False, root_only=False, recordTime=True):
    # Start timing overall execution
    start_process = time.time() if recordTime else None
    times = {} if recordTime else None

    ## if the text is too long, we try to trim it to the last whitespace
    start_time = time.time() if recordTime else None
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
        transliterated_text = anythingToSLP1(text)     
        text = regex.sub('[^\p{L}\']', '', transliterated_text)
        if recordTime:
            times['transliteration'] = time.time() - start_time

        start_time = time.time() if recordTime else None
        result = root_any_word(text)
        if recordTime:
            times['root_analysis'] = time.time() - start_time

        start_time = time.time() if recordTime else None
        if result is not None:
            for i, res in enumerate(result):
                if isinstance(res, str):
                    result[i] = transliterateSLP1IAST(res.replace('-', ''))
                elif isinstance(res, list):
                    if isinstance(res[0], str):
                        res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
            result_vocabulary = get_voc_entry(result, *dict_names)
            if recordTime:
                times['dictionary_lookup'] = time.time() - start_time

            return clean_results(result_vocabulary, debug=debug, root_only=root_only)
        else:
            query = [transliterateSLP1IAST(text)]
            result_vocabulary = get_voc_entry(query, *dict_names)  
            if isinstance(result_vocabulary[0][2], dict):
                if recordTime:
                    times['total_time'] = time.time() - start_process
                return clean_results(result_vocabulary, debug=debug, root_only=root_only)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    start_time = time.time() if recordTime else None
    text = transliterateSLP1IAST(text)
    splitted_text = hybrid_sandhi_splitter(text)
    if recordTime:
        times['sandhi_splitting'] = time.time() - start_time

    start_time = time.time() if recordTime else None
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
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


##process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

filtered_words = ["ca", "na", "eva", "ni", "apya", "ava", "sva"]


def clean_results(list_of_entries, root_only=False, debug=True):

    i = 0
    if debug == True:
        print("it breaks right here:", list_of_entries)

    #print("is it broken here?", list_of_entries[i])

    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        # Check if the word is in filtered_words
        if list_of_entries[i][0] in filtered_words:
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == list_of_entries[i][0]:
                del list_of_entries[i + 1]

        if list_of_entries[i][0] == "duḥ" and list_of_entries[i+1][0] == "kha":
            replacement = get_voc_entry(["duḥkha"])
            if replacement is not None:
                list_of_entries[i] = replacement[0]
                del list_of_entries[i + 1]
                if list_of_entries[1+2] == "kha":
                    del list_of_entries[i + 2]  ##it's kha also as well

        if len(list_of_entries[i]) >= 5 and list_of_entries[i][0][-1] == "n" and list_of_entries[i][4] != list_of_entries[i][0]:
            #print("the one not replaced:", list_of_entries[i])
            if list_of_entries[i][4] in mwdictionaryKeys:
                replacement = get_voc_entry([list_of_entries[i][4]])
                if replacement is not None:
                    list_of_entries[i] = replacement[0]
        

        
        # Check if the word is "sam"
        if list_of_entries[i][0] == "sam":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "sa" or list_of_entries[j][0] == "sam"):
                j += 1
            if j < len(list_of_entries):


                                ##non ha senso CHECK IF sam or sam  + list_of_entries[j][0]] are in MW dict
                                ## a quel punto fai voc_entry di quello, e rimpiazza tutte le entry inutili.

                voc_entry = get_voc_entry(["sam" + list_of_entries[j][0]])
                #print("voc_entry", voc_entry)

                ##non ha senso
                
                # Ensure voc_entry is not None and has the expected structure
                if (voc_entry is not None and len(voc_entry) > 0 and 
                    isinstance(voc_entry[0], list) and len(voc_entry[0]) > 2 and 
                    isinstance(voc_entry[0][2], dict) and 'MW' in voc_entry[0][2]):
                    
                    # Check if the first key of the dictionary inside MW matches the condition
                    first_key = next(iter(voc_entry[0][2]['MW']), None)
                    if first_key and voc_entry[0][0] == first_key:
                        print("revised query", ["saṃ" + list_of_entries[j][0]])
                        voc_entry = get_voc_entry("saṃ" + list_of_entries[j][0])
                        print("revise_voc_entry", voc_entry)
        
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "anu"
        if list_of_entries[i][0] == "anu":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "anu"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["anu" + list_of_entries[j][0]])
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "ava"
        if list_of_entries[i][0] == "ava":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "ava"):
                j += 1
            if j < len(list_of_entries):
                print("testing with:", ["ava" + list_of_entries[j + 1][0]])
                voc_entry = get_voc_entry(["ava" + list_of_entries[j + 1][0]])
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]        
        i += 1  
    
    print("list of roots:")

    def roots(list_of_entries, debug=debug):
        roots = []
        for entry in list_of_entries:
            if not roots or roots[-1] != entry[0]:
                if debug==True:
                    print(entry[0])
                roots.append(entry[0])
        return roots
    
    if root_only==True: 
        return roots(list_of_entries, debug)
        
    return list_of_entries

## hard mode testing:
#process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")







#dict_names = ["AE", "AP90", "MW", "MWE"]
#path = "/resources/Sanskrit_dictionaries/"

def multidict(name: str, *args: str, source: str = "MW") -> Dict[str, Dict[str, List[str]]]:
    dict_names: List[str] = []
    dict_results: Dict[str, Dict[str, List[str]]] = {}
    name_component: str = ""
    
    # Collect dictionary names
    if not args:
        dict_names.append(source)
    else:
        dict_names.extend(args)
    
    session = Session()
    
    # For each dictionary, perform queries and process results
    for dict_name in dict_names:
        
        
        # Initial query
        query_builder = f"""
        SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
        WHERE keys_iast = :name 
        OR keys_iast LIKE :wildcard_name
        """
        wildcard_name = f"{name}"
        
        with db.engine.connect() as connection:

            results = connection.execute(
                text(query_builder), 
                {"name": name, "wildcard_name": wildcard_name}
            ).fetchall()

        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
            WHERE keys_iast = :name 
            OR keys_iast LIKE :wildcard_name
            """
            wildcard_name = f"{name[:-1]}_"
            
            with db.engine.connect() as connection:

                results = connection.execute(
                    text(query_builder), 
                    {"name": name, "wildcard_name": wildcard_name}
                ).fetchall()
        
        #print(f"Results for {dict_name}: {results}")
        
        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
            WHERE keys_iast LIKE :name1 
            OR keys_iast LIKE :name2
            """
            with db.engine.connect() as connection:

                results = connection.execute(
                    text(query_builder), 
                    {"name1": name + "_", "name2": name[:-1] + "_"}
                ).fetchall()

        #print(f"Results for {dict_name} after second query: {results}")
        

        # Group results by components
        component_dict: Dict[str, List[str]] = {}
        for row in results:
            #print(f"Row: {row}")
            key_iast, components, cleaned_body = row
            if not name_component:
                name_component = components
            #print(f"key_iast: {key_iast}, components: {components}, cleaned_body: {cleaned_body}")
            if key_iast in component_dict:
                component_dict[key_iast].append(cleaned_body)
            else:
                component_dict[key_iast] = [cleaned_body]        
        # Add to dict_results
        dict_results[dict_name] = component_dict
    
    connection.close()
    return [name_component, dict_results]



# Example usage
#results = multidict("yoga", "MW" "AP90")
#print(results)





def get_mwword(word:str)->list[str, str, list[str]] : 
        session = Session()
        query_builder = text("SELECT components, cleaned_body FROM mwclean WHERE keys_iast = :word")
        print("query_builder", query_builder, word)
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]



def get_voc_entry(list_of_entries, *dict_names):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]

            if '*' not in word and '_' not in word and '%' not in word:
                if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    print("word", word)
                    entry = entry + multidict(word, *dict_names)
                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                entries.append(entry)
            else:
                print("wildcard search", word)
                entry = entry + multidict(word, *dict_names)
                entries.append(entry)
            
        elif isinstance(entry, str):
            
            if '*' not in entry and '_' not in entry and '%' not in entry:
                if entry in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    print("word", entry)
                    entry = [entry] + multidict(entry, *dict_names)
                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                entries.append(entry)
            else:
                print("wildcard search", entry)
                entry = [entry] + multidict(entry, *dict_names)
                entries.append(entry)
    return entries



# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component

