import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import ast
from detectTransliteration import detect
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



__version__ = "0.1"
def print_version():
    print(f"Version: {__version__}")


import os
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

DATABASE_URL = "sqlite:///resources/merged_formdb.sqlite"

logging.basicConfig(level=logging.CRITICAL)

##to get all the available schemes
##indic_transliteration.sanscript.SCHEMES.keys()
    
def transliterateSLP1IAST(text):
    return transliterate(text, sanscript.SLP1, sanscript.IAST)   
def transliterateIASTSLP1(text):
    return transliterate(text, sanscript.IAST, sanscript.SLP1)   


def transliterateSLP1HK(text):
    return transliterate(text, sanscript.SLP1, sanscript.HK)   

def transliterateDEVSLP1(text):
    return transliterate(text, sanscript.DEVANAGARI, sanscript.SLP1)
        
def anythingToSLP1(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.SLP1)
def anythingToIAST(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.IAST)
def anythingToHK(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.HK)

def transliterateAnything(text, transliteration_scheme):
    detected_scheme_str = detect(text).upper()
    transliteration_scheme_str = transliteration_scheme.upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    output_scheme = getattr(sanscript, transliteration_scheme_str)
    return sanscript.transliterate(text, detected_scheme, output_scheme)

##qui la lista degli encoding è lowercase


parser = Parser(output_encoding='iast')

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
# Create a session
session = Session()

Base = declarative_base()

# Define the split_cache model
class SplitCache(Base):
    __tablename__ = 'split_cache'
    
    input = Column(String, primary_key=True)
    splitted_text = Column(String)

# Create the table if it doesn't exist
Base.metadata.create_all(engine)


def sandhi_splitter(text_to_split):
    """
    Splits the given text using a sandhi splitter parser.
    Checks if the result is already cached before performing the split.

    Parameters:
    - text_to_split (str): The text to split.

    Returns:
    - list: A list of split parts of the text.
    """

    #text_to_split = anythingToSLP1(text_to_split).strip()

    # Check if the result is already cached
    cached_result = session.query(SplitCache).filter_by(input=text_to_split).first()
    
    if cached_result:
        # Retrieve and return the cached result if it exists
        splitted_text = ast.literal_eval(cached_result.splitted_text)
        print(f"Retrieved from cache: {splitted_text}")
        return splitted_text

    # If not cached, perform the split
    try:
        splits = parser.split(text_to_split, limit=1)

        #if split is none, default to split by space
        if splits is None:
            return text_to_split.split()
        for split in splits:
            splitted_text = f'{split}'
        splitted_text = ast.literal_eval(splitted_text)

        print(f"Splitted text: {splitted_text}")

        # Store the split result in cache as a list
        new_cache_entry = SplitCache(input=text_to_split, splitted_text=str(splitted_text))
        session.add(new_cache_entry)
        session.commit()
        print(f"Added to cache: {splitted_text}")

        return splitted_text

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        return text_to_split.split()




stopwords = pd.read_csv('resources/stopwords.csv')

stopwords_as_list = stopwords['stopword'].tolist()

def remove_stopwords_list(text_list):
    return [word for word in text_list if word not in stopwords_as_list]

def remove_stopwords_string(text):
    text = text.replace('.', '')  # Remove periods
    text_list = text.split()  # Split the string into words
    return ' '.join(word for word in text_list if word not in stopwords_as_list)


with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

with open('resources/MWKeysOnlySLP1.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeysSLP1 = json.load(f)


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
                query_builder2 = "SELECT * FROM lgtab1 WHERE stem = :root_form and model = :type "
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
            query_builder = "SELECT * FROM vlgtab2 WHERE key = :verb"
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
                query_builder2 = "SELECT * FROM vlgtab1 WHERE stem = :root_form and model = :type"
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

def root_any_word(word):


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




def root_any_list(text_list):
    roots = []
    for word in text_list:
        root_found = root_any_word(word)
        if root_found is not None:
            roots.append(root_found)
    
    # Transliterate the first element of each tuple
    for i in range(len(roots)):
        roots[i] = (transliterateSLP1IAST(roots[i][0].replace('-', '')),) + roots[i][1:]
    
    return roots


def dict_word_iterativeIAST(word):
    sanskrit_end_letters = ['a', 'ā', 'i', 'ī', 'u', 'ū', 'e', 's']  # Add all Sanskrit end-of-word letters here
    temp_word = word
    removed_part = ''  # Part of the word that was removed
    found = False  # Flag to indicate when a match has been found
    if temp_word in mwdictionaryKeys:  
            found_word = temp_word 
            found = True
            return [found_word, removed_part]
    while temp_word and not found: # Continue the loop until a match is found or temp_word is empty        
        if temp_word[:-1] in mwdictionaryKeys:  
            found_word = temp_word[:-1] 
            #print(temp_word[:-1] ) debug
            found = True
            return [found_word, removed_part]       
        elif temp_word[-1].isalpha():  # If the last character is a letter
            for letter in sanskrit_end_letters:
                attempt = temp_word[:-1] + letter
                #print(attempt) debug
                if attempt in mwdictionaryKeys:
                    found_word = attempt
                    found = True  # Set the flag to True when a match is found
                    break                                
        removed_part = temp_word[-1] + removed_part  # Keep track of the removed part
        temp_word = temp_word[:-1]  # Remove the last character, regardless of whether it's a letter or not
    
    return [found_word, removed_part] if found else None  # Return the match if found, else return None

def dict_word_iterative(word):
    sanskrit_end_letters = ["a", "A", "i", "I", "u", "U", "f", "d", "n", "t", "m", "s", "h"]  # Add all Sanskrit end-of-word letters here
    temp_word = word
    removed_part = ''  # Part of the word that was removed
    found = False  # Flag to indicate when a match has been found
    if temp_word in mwdictionaryKeysSLP1:  
            found_word = temp_word 
            found = True
            return [found_word, removed_part]
    while temp_word and not found: # Continue the loop until a match is found or temp_word is empty        
        if temp_word[:-1] in mwdictionaryKeysSLP1:  
            found_word = temp_word[:-1] 
            #print(temp_word[:-1] ) debug
            found = True
            return [found_word, removed_part]       
        elif temp_word[-1].isalpha():  # If the last character is a letter
            for letter in sanskrit_end_letters:
                attempt = temp_word[:-1] + letter
                #print(attempt) debug
                if attempt in mwdictionaryKeysSLP1:
                    found_word = attempt
                    found = True  # Set the flag to True when a match is found
                    break                                
        removed_part = temp_word[-1] + removed_part  # Keep track of the removed part
        temp_word = temp_word[:-1]  # Remove the last character, regardless of whether it's a letter or not
    
    return [found_word, removed_part] if found else None  # Return the match if found, else return None




        ##if the dictionary approach fails, try the iterative approach:


def root_compounds(word):

    if word[0] == "'":
        word = 'a' + word[1:]

    first_root = dict_word_iterative(word)
    #print("first_root", first_root)
    first_root_entry = root_any_word(first_root[0])
    #print("first_root_entry", first_root_entry)
    
    ## if it's a compound
    if first_root is not None and first_root[1] is not None and len(first_root[1]) >= 4:
        
        ##remove the first root from the word
        without_root = word.replace(first_root[0], '', 1)  # Only replace the first occurrence
        
        ##try the dictionary approach
        second_root = root_any_word(without_root)
        
        ##if the dictionary approach fails, try the iterative approach:
        if second_root == None:
            second_root = dict_word_iterative(without_root)
            #print("second_root", second_root)
            if second_root is None:
                if first_root_entry is not None:
                    return first_root_entry
                else:
                    return [first_root[0]]
            if len(second_root[0]) < 2:
                second_root = None
            else:
                second_root_try = root_any_word(second_root[0])
                if second_root_try is not None:
                    second_root = second_root_try
                else: 
                    second_root = [second_root[0]]            
            if second_root is not None:
                if first_root_entry is not None:                    
                    return first_root_entry + second_root
                else:
                    return [first_root[0]] + second_root
            else:
                if first_root_entry is not None:
                    return first_root_entry
                else:
                    return [first_root[0]]
        else:
            if first_root_entry is not None:
                return first_root_entry + second_root
            else:
                return [first_root[0]] + second_root
            
    ## if it's not a compound        
    else:
        if first_root_entry is not None:
            return first_root_entry
        else:
            return [first_root[0]]
            

def root_compounds(word):
    if word.startswith("'"):
        word = 'a' + word[1:]

    # Base case: if word is empty, return empty list
    if not word:
        return []

    roots = []

    first_root = dict_word_iterative(word)
    if not first_root or not first_root[0]:
        # No root found
        return []
    print("first_root", first_root)
    first_root_entry = root_any_word(first_root[0])
    print("first_root_entry", first_root_entry)
    if first_root_entry is not None:
        if isinstance(first_root_entry, list):
            roots.extend(first_root_entry)
        else:
            roots.append(first_root_entry)
    else:
        roots.append(first_root[0])

    # Remove the first root from the word
    without_root = word.replace(first_root[0], '', 1)

    # If there's nothing left, return the roots found
    if not without_root:
        return roots

    # Check if the word is being reduced
    if len(without_root) >= len(word):
        # Word not reduced; cannot proceed further
        return roots

    # Now, recursively process the remaining word
    rest_entries = root_any_word(without_root)
    if rest_entries is not None:
        if isinstance(rest_entries, list):
            roots.extend(rest_entries)
        else:
            roots.append(rest_entries)
    else:
        # Try dict_word_iterative and ensure result is longer than 2
        rest_root = dict_word_iterative(without_root)
        if rest_root is None or len(rest_root[0]) < 2:
            # Cannot proceed further; return roots found so far
            return roots
        else:
            # Check again if the word is being reduced
            if len(without_root) >= len(word):
                return roots
            # Recursively process the rest of the word
            rest_roots = root_compounds(without_root)
            roots.extend(rest_roots)

    return roots





def inflect(splitted_text):
    roots = []
    prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'abhi', 'ni', 'pari', 'prati', 'parA', 'ava', 'adhi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duḥ']  # Add more prefixes as needed
    i = 0
    while i < len(splitted_text):
        word = splitted_text[i]
        if word in prefixes and i + 1 < len(splitted_text):
            next_word = splitted_text[i + 1]
            if word == 'sam':
                combined_words = ['sam' + next_word, 'saM' + next_word]
            elif word == 'vi':
                combined_words = ['vi' + next_word, 'vy' + next_word]
            else:
                combined_words = [word + next_word]

            rooted = None
            for combined_word in combined_words:
                rooted = root_any_word(combined_word)
                if rooted is not None:
                    break  # Exit loop if a valid root is found

            if rooted is not None:
                roots.extend(rooted)
                i += 2  # Skip next word since it's part of the combined word
                continue
            else:
                rooted_word = root_any_word(word)
                if rooted_word is not None:
                    roots.extend(rooted_word)
                else:
                    compound_try = root_compounds(word)
                    if compound_try is not None:
                        roots.extend(compound_try)
                    else:
                        roots.append(word)
                i += 1  # Move to next word
        else:
            rooted = root_any_word(word)
            if rooted is not None:
                roots.extend(rooted)
            else:
                compound_try = root_compounds(word)
                if compound_try is not None:
                    roots.extend(compound_try)
                else:
                    roots.append(word)
            i += 1

    # Transliterate roots
    for j in range(len(roots)):
        if isinstance(roots[j], list):
            roots[j][0] = transliterateSLP1IAST(roots[j][0].replace('-', ''))
        else:
            roots[j] = transliterateSLP1IAST(roots[j].replace('-', ''))
    return roots


## bug with process("nīlotpalapatrāyatākṣī")    





def process(text, *dict_names):


    ## if the text end with a *, remove it and try to find the word in the dictionary for exact match

    if text[-1] == '*':
        voc_entry = get_voc_entry([anythingToIAST(text[:-1])], *dict_names)
        if voc_entry is not None:
            return voc_entry
        else:
            process(text[:-1])

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:
        #print("single_word", text)
        transliterated_text = anythingToSLP1(text)     
        #print("transliterated_text", transliterated_text)
        text = regex.sub('[^\p{L}\']', '', transliterated_text)
        ## here it should be transliterated to SLP1 before and added aH at the end instead of a
        if text[0] == "'":
            text = 'a' + text[1:]
        if text[-1] == 'o':
            text = text[:-1] + 'aH'
        elif text[-1] == 'S':
            text = text[:-1] + 'H'
        elif text[-1] == 'y':
            text = text[:-1] + 'i'
        #print("text", text)
        if "o'" in text:
            modified_text = re.sub(r"o'", "aH a", text)
            #print("modified_text", modified_text)
            result = process(modified_text)
            return result
        result = root_any_word(text)
        if result is None and text[-1] == 'M':
            text = text[:-1] + 'm'
            result = root_any_word(text)
        if result is None and text[0:1] == "ch":
            text = 'S' + text[2:] 
        if result is not None:
            for res in result:
                if isinstance(res, list):
                    res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
                    #print("res", res)
            result_vocabulary = get_voc_entry(result, *dict_names)  
            print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary)
        else:
            query = [transliterateSLP1IAST(text)]
            print("query", query)
            result_vocabulary = get_voc_entry(query, *dict_names)  
            print("result_vocabulary", result_vocabulary)
            if result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
            
    ## attempt to remove sandhi and tokenise in any case
    #print("pre_splitted_text", text)
    text = transliterateSLP1IAST(text)
    #print("transliterate to split", text)
    splitted_text = sandhi_splitter(text)
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary)


##process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

filtered_words = ["ca", "na", "eva", "ni", "apya", "ava", "sva"]


def clean_results(list_of_entries):

    i = 0
    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        # Check if the word is in filtered_words
        if list_of_entries[i][0] in filtered_words:
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == list_of_entries[i][0]:
                del list_of_entries[i + 1]
        
        # Check if the word is "sam"
        if list_of_entries[i][0] == "sam":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "sa" or list_of_entries[j][0] == "sam"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["sam" + list_of_entries[j][0]])
                print("voc_entry", voc_entry)
                if voc_entry[0][0] == voc_entry[0][2][0]:
                    print("revised query", ["saM" + list_of_entries[j][0]])
                    voc_entry = process("saM" + list_of_entries[j][0])
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
    return list_of_entries

def process_test(text, remove_stopwords = False, dictionary_entry = True, output_encoding = "IAST", entry_only = False):
    
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)
    
    ## removes stopwords
    if remove_stopwords == True: 
        splitted_text = remove_stopwords_list(splitted_text)
        
    
    inflections = inflect(splitted_text) 
    
    if dictionary_entry == True:
        inflections = get_voc_entry(inflections)    

    if entry_only == True:
        entry_list = []
        for entry in inflections:
            entry_list.append(entry[0])
        inflections = entry_list    

    return clean_results(inflections)

## hard mode testing:
#process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")



def preprocess(text):

    remove_char_text = ''.join(c for c in text if c.isalpha() or c == "'" or c == ' ')

    print("processing", text)
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(remove_char_text)   

    splitted_text = remove_stopwords_list(splitted_text)    
    
    inflections = inflect(splitted_text) 
    
    entry_list = []
    entry_list = [entry[0] for entry in inflections if len(entry[0]) > 1]
    entry_list = [key for key, group in groupby(entry_list)]

    i = 0
    while i < len(entry_list) - 1:
        # Normalize words to form without diacritics
        word1 = unicodedata.normalize('NFD', entry_list[i])
        word2 = unicodedata.normalize('NFD', entry_list[i+1])

        # Remove diacritics
        word1 = ''.join(c for c in word1 if not unicodedata.combining(c))
        word2 = ''.join(c for c in word2 if not unicodedata.combining(c))

        if word1 in word2 or word2 in word1:
            if len(entry_list[i]) > len(entry_list[i+1]):
                entry_list.pop(i)
            else:
                entry_list.pop(i+1)
        else:
            i += 1
    print("processed", entry_list)
    return clean_results(entry_list)



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
        wildcard_name = f"{name[:-1]}_"
        
        with engine.connect() as connection:
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
            with engine.connect() as connection:
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



def get_voc_entry(list_of_entries, *args: str, source: str = "MW"):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):
            
            word = entry[0]
            dict_entry = []
            if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                key2 = dictionary_json[word][0][1] ##qui si dovrebbe fare un fetch SQL 
                key2 = transliterateSLP1IAST(key2) ## qui ho un leggero problema, se prendo l'equivalente in SQL da dove prendo la key 2? dalla prima o dalla seconda voce. Possibilmente dalla seconda se c'è più di una voce. 
    
                for entry_dict in dictionary_json[word]:
                    dict_entry.append(re.sub(r'<s>(.*?)</s>', lambda m: '<s>' + transliterateSLP1IAST(m.group(1)) + '</s>', entry_dict[4]))
                entry.append(key2)
                entry.append(dict_entry)
            else:
                entry.append(word)  # Append the original word for key2
                entry.append([word])  # Append the original word for dict_entry
            entries.append(entry)
            
        elif isinstance(entry, str):
            
            dict_entry = []
            if entry in mwdictionaryKeys:  # Check if the key exists
                key2 = dictionary_json[entry][0][1]
                key2 = transliterateSLP1IAST(key2)
    
                for entry_dict in dictionary_json[entry]:
                    dict_entry.append(re.sub(r'<s>(.*?)</s>', lambda m: '<s>' + transliterateSLP1IAST(m.group(1)) + '</s>', entry_dict[4]))
                entry = [entry]    
                
                entry.append(key2)
                entry.append(dict_entry)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
    return entries

# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component


def get_mwword(word:str)->list[str, list[str]] : 
        session = Session()
        query_builder = "SELECT key2, cleaned_body FROM mwclean WHERE keys_iast = :word"
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]




def get_voc_entry(list_of_entries):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]

            if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                entry = entry + get_mwword(word)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
            
        elif isinstance(entry, str):
            
            if entry in mwdictionaryKeys:  # Check if the key exists
                entry = [entry] + get_mwword(entry)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
    return entries

# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component


def get_voc_entry(list_of_entries, *dict_names):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]

            if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                entry = entry + multidict(word, *dict_names)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
            
        elif isinstance(entry, str):
            
            if entry in mwdictionaryKeys:  # Check if the key exists
                entry = [entry] + multidict(word, *dict_names)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
    return entries