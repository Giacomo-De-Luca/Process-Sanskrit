import indic_transliteration

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

import ast
from detectTransliteration import detect

import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug('This is a debug message')
logging.info('This is an informational message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

##to get all the available schemes
##indic_transliteration.sanscript.SCHEMES.keys()
    
def transliterateSLP1IAST(text):
    return transliterate(text, sanscript.SLP1, sanscript.IAST)   

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

from sanskrit_parser import Parser

parser = Parser(output_encoding='iast')

def sandhi_splitter(text_to_split):
    try:
        splits = parser.split(text_to_split, limit=1)
        if splits is None:
            return text_to_split.split()
        for split in splits:
            splitted_text = f'{split}'
        splitted_text = ast.literal_eval(splitted_text)
        return splitted_text
    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        return text_to_split.split()
    

import pandas as pd

stopwords = pd.read_csv('resources/stopwords.csv')

stopwords_as_list = stopwords['stopword'].tolist()

def remove_stopwords_list(text_list):
    return [word for word in text_list if word not in stopwords_as_list]

def remove_stopwords_string(text):
    text = text.replace('.', '')  # Remove periods
    text_list = text.split()  # Split the string into words
    return ' '.join(word for word in text_list if word not in stopwords_as_list)


import json
import sqlite3
import re

with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

##given a name finds the root

def SQLite_find_name(name):
    
    original_word = name    
    # Transliterate input
    query_transliterate = anythingToSLP1(name)    
    # FormatSQL query

    outcome = [] 

    def query1(word):
        query_builder = "SELECT * FROM lgtab2 WHERE key = ?"  
        ##OpenConnection, make SQL query to find the root    
        sqliteConnection = sqlite3.connect('resources/SQLite_db/lgtab2.sqlite')
        cursor = sqliteConnection.cursor()
        cursor.execute(query_builder, (word,))
        results = cursor.fetchall()
        sqliteConnection.close()
        ## da riflettere meglio su come voglio fare su questo... se ci sono risultati duplicati è perché un nome può essere maschile neutro e femminile, in tal caso voglio che risulti, senza duplicare le entrate del dizionario etc. 
        ##remove duplicate results:
        #results_dict = {t[2]: t for t in results}    
        # Convert the dictionary back to a list of tuples
        #results = list(results_dict.values())
        return results
    
    results = query1(query_transliterate)
    
    if not results:  # If query1 didn't find any results
        if query_transliterate[-1] == 'M':
            query_transliterate = query_transliterate[:-1] + 'm'
            results = query1(query_transliterate)
    
    for inflected_form, type, root_form in results: 
        #print("result", (inflected_form, type, root_form))
        ## get root, inflected form, type (could be useful later)
        if not root_form:  # If root_form is None or empty
            return  # End the function
        inflected_form_var = inflected_form
        type_var = type
        root_form_var = root_form  
        ##break the function if it's not a name; important! 
        if not root_form:  # If root_form is still None after the loop
            return  # End the function
        
        ## build query in the second table to find the inflection table
        query_builder2 = "SELECT * FROM lgtab1 WHERE stem = ? and model = ?"
        sqliteConnection = sqlite3.connect('resources/SQLite_db/lgtab1.sqlite')
        cursor = sqliteConnection.cursor()
        cursor.execute(query_builder2, (root_form, type_var))
        result = cursor.fetchall()
        sqliteConnection.close()

        print("word_result", result)

        #print("result", result)

        #print("refs", result[0][2])

        ##word_refs = re.match(r"\d+,([a-zA-Z]+)", result[0][2]).group(1)
        word_refs = re.findall(r",([a-zA-Z]+)",result[0][2])[0]

        ## get the inflection table as a list of words instead of tuple

        inflection_tuple = result[0][3]  # Get the first element of the first tuple
        #print("inflection_tuple", inflection_tuple)
        inflection_words = inflection_tuple.split(':') 
        #print("inflection_words", inflection_words)

        ##transliterate back the result to IAST for readability

        inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
        query_transliterateIAST = transliterateSLP1IAST(query_transliterate)

        ##make Inflection Table

        indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]
        # Define row and column titles
        rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
        coltitles = ["Sg", "Du", "Pl"]

        from tabulate import tabulate

        # Your list of strings

        # Split the list into a 6x3 table
        #table = [inflection_wordsIAST[i:i+3] for i in range(0, len(inflection_wordsIAST), 3)]

        # Highlight the matched cells with red color
        #table = [["\033[31m" + cell + "\033[0m" if i*3 + j in indices else cell for j, cell in enumerate(row)] for i, row in enumerate(table)]

        # Add row titles to the table
        #table = [[rowtitle] + row for rowtitle, row in zip(rowtitles, table)]

        # Print the table with column titles
        #print(tabulate(table, headers=coltitles, tablefmt="grid"))

        # Print the indices of the query in the list
        if indices:
            # Convert the indices to row and column names
            row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
        else: 
            row_col_names = None
        
        outcome.append([word_refs, type_var, row_col_names, inflection_wordsIAST, original_word])
    return outcome



def SQLite_find_verb(verb):
    
    original_verb = verb
    
    # Transliterate input
    query_transliterate = anythingToSLP1(verb)
    
    # FormatSQL query
    query_builder = "SELECT * FROM vlgtab2 WHERE key = ?"
    
    ##OpenConnection, make SQL query to find the root
    
    sqliteConnection = sqlite3.connect('resources/SQLite_db/vlgtab2.sqlite')
    cursor = sqliteConnection.cursor()
    #"select * from `$table` where `key`=\"$key\"";
    cursor.execute(query_builder, (query_transliterate,))
    result = cursor.fetchall()
    sqliteConnection.close()
    #print("result1", result)
    root_form = None
    
    ## get root, inflected form, type (could be useful later)
    
    for inflected_form, type, root_form in result:
        if not root_form:  # If root_form is None or empty
            return  # End the function
        inflected_form_var = inflected_form
        type_var = type
        root_form_var = root_form   
       # print(f"Inflected Form: {inflected_form}, Type: {type}, Root Form: {root_form}")
    
    if not root_form:  # If root_form is None or empty
            return  # End the function
        
    ## build query in the second table to find the inflection table
            
    query_builder2 = "SELECT * FROM vlgtab1 WHERE stem = ? and model = ?"

    sqliteConnection = sqlite3.connect('resources/SQLite_db/vlgtab1.sqlite')

    cursor = sqliteConnection.cursor()
    #print('DB Init')
    cursor.execute(query_builder2, (root_form, type_var))
    result = cursor.fetchall()
    sqliteConnection.close()
    #print("result2:", result)
    
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
    query_transliterateIAST = transliterateSLP1IAST(query_transliterate)
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]

    # Define row and column titles
    rowtitles = ["First", "Second", "Third"]
    coltitles = ["Sg", "Du", "Pl"]

    from tabulate import tabulate

    # Your list of strings

    # Split the list into a 6x3 table
    #table = [inflection_wordsIAST[i:i+3] for i in range(0, len(inflection_wordsIAST), 3)]

    # Highlight the matched cells with red color
    #table = [["\033[31m" + cell + "\033[0m" if i*3 + j in indices else cell for j, cell in enumerate(row)] for i, row in enumerate(table)]

    # Add row titles to the table
    #table = [[rowtitle] + row for rowtitle, row in zip(rowtitles, table)]

    # Print the table with column titles
    #print(tabulate(table, headers=coltitles, tablefmt="grid"))

    # Print the indices of the query in the list
    if indices:
        # Convert the indices to row and column names
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
        #print(f"The row and column names of '{query_transliterateIAST}' are {row_col_names}.")
    #else:
       # print(f"'{query_transliterateIAST}' is not in the list.")
    else:
        row_col_names = None
        
    return [[stem, type_var, row_col_names, inflection_wordsIAST, original_verb]]



## also map to the type.

import pandas as pd

# Read the Excel file into a DataFrame
type_map = pd.read_excel('resources/type_map.xlsx')

def root_any_word(word):
    result_roots = SQLite_find_name(word)
    if not result_roots:  # If process_word didn't find any results
        result_roots = SQLite_find_verb(word)

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


def dict_word_iterative(word):
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



        ##if the dictionary approach fails, try the iterative approach:


def root_compounds(word):
    
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
            


def inflect(splitted_text):
    roots = []
    
#    print("splitted", splitted_text)
    for word in splitted_text: 
        
        rooted = root_any_word(word)
        if rooted is not None:
            for root in rooted:
                roots.append(root)  
        else:
            compound_try = root_compounds(word)
            if compound_try is not None:
                roots.extend(compound_try)  
                #print("compound_try", compound_try)
            else:
                roots.extend(word)  
                
    for i in range(len(roots)):
        if isinstance(roots[i], list):
            #print("debug", roots)
            #print(roots[i][0])
            roots[i][0] = transliterateSLP1IAST(roots[i][0].replace('-', ''))
        else:
            roots[i] = transliterateSLP1IAST(roots[i].replace('-', ''))           
    print("inflect roots", roots)
    return roots             

## bug with process("nīlotpalapatrāyatākṣī")    


with open('resources/no_abbreviationMW.json') as f:
    # Load JSON data from file
    dictionary_json = json.load(f)


def get_voc_entry(list_of_entries):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):
            
            word = entry[0]
            dict_entry = []
            if word in dictionary_json:  # Check if the key exists
                key2 = dictionary_json[word][0][1]
                key2 = transliterateSLP1IAST(key2)
    
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
            if entry in dictionary_json:  # Check if the key exists
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

import regex

def process(text):

    if ' ' not in text:
        #print("single_word", text)
        transliterated_text = anythingToHK(text)     
        #print("transliterated_text", transliterated_text)
        text = regex.sub('[^\p{L}\']', '', transliterated_text)
        ## here it should be transliterated to SLP1 before and added aH at the end instead of a
        if text[-1] == 'o':
            text = text[:-1] + 'aH'
        result = root_any_word(text)
        if result is None and text[-1] == 'M':
            text = text[:-1] + 'm'
            result = root_any_word(text)
        if result is not None:
            for res in result:
                if isinstance(res, list):
                    res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
            result_vocabulary = get_voc_entry(result)  
            return clean_results(result_vocabulary)
            
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)    
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary)


##process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

def clean_results(list_of_entries):
    i = 0
    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        if list_of_entries[i][0] == "apya" and list_of_entries[i + 1][0] == "ap":
            list_of_entries[i] = [item for sublist in get_voc_entry(["api"]) for item in sublist]
            del list_of_entries[i + 1]
        elif list_of_entries[i][0] == "ca":
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == "ca":
                del list_of_entries[i + 1]
        elif list_of_entries[i][0] == "eva":
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == "eva":
                del list_of_entries[i + 1]
        elif list_of_entries[i][0] == "sam":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "sa" or list_of_entries[j][0] == "sam"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["sam" + list_of_entries[j][0]])
                if voc_entry is None:
                    voc_entry = get_voc_entry(["saM" + list_of_entries[j][0]])
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

    return inflections
## hard more testing:
#process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

def preprocess(text):
    
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)

    splitted_text = remove_stopwords_list(splitted_text)    
    
    inflections = inflect(splitted_text) 
    
    entry_list = []
    for entry in inflections:
        entry_list.append(entry[0])
        inflections = entry_list    

    return inflections