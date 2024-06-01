import sanskrit_parser as sp
import indic_transliteration

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from sanskrit_parser import Parser

from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import ast
from detectTransliteration import detect



indic_transliteration.sanscript.SCHEMES.keys()


def transliterateVELTIAST(text):
     return transliterate(text, sanscript.VELTHUIS, sanscript.IAST)


def transliterateIASTDEV(text):
     return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)

def transliterateDEVIAST(text):
    return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)

def transliterateIASTSPL1(text):
     return transliterate(text, sanscript.IAST, sanscript.SLP1)

def transliterateHKSPL1(text):
     return transliterate(text, sanscript.HK, sanscript.SLP1)    
    
def transliterateSLP1IAST(text):
     return transliterate(text, sanscript.SLP1, sanscript.IAST)    
        
    
def transliterateHKIAST(text):
     return transliterate(text, sanscript.HK, sanscript.IAST)        
    
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

transliterateIASTSPL1("śunyānām")


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

with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

import sqlite3

##given a name finds the root
import sqlite3

##given a name finds the root

def SQLite_find_name(name):
    
    original_word = name
    
    # Transliterate input
    query_transliterate = anythingToSLP1(name)
    
    # FormatSQL query
    #query_builder = "SELECT * FROM lgtab2 WHERE key = " + "'" + query_transliterate + "'"
    query_builder = "SELECT * FROM lgtab2 WHERE key = ?"        
    
    ##OpenConnection, make SQL query to find the root
    
    sqliteConnection = sqlite3.connect('SQLite_db/lgtab2.sqlite')

    cursor = sqliteConnection.cursor()
   # print('DB Init')

    #"select * from `$table` where `key`=\"$key\"";
    cursor.execute(query_builder, (query_transliterate,))
    result = cursor.fetchall()
    #print('SQLite Version is {}'.format(result))
    
    ## get root, inflected form, type (could be useful later)
    
    root_form = None
    
    for inflected_form, type, root_form in result:
        if not root_form:  # If root_form is None or empty
            return  # End the function
        inflected_form_var = inflected_form
        type_var = type
        root_form_var = root_form   
        #print(f"Inflected Form: {inflected_form}, Type: {type}, Root Form: {root_form}")
    
    ##break the function if it's not a verb; important! 
        
    if not root_form:  # If root_form is still None after the loop
        return  # End the function
    
    
    ## build query in the second table to find the inflection table
    
    query_builder2 = "SELECT data FROM lgtab1 WHERE stem = ?"
    
    sqliteConnection = sqlite3.connect('SQLite_db/lgtab1.sqlite')

    cursor = sqliteConnection.cursor()
    #print('DB Init')

    cursor.execute(query_builder2, (root_form,))
    result = cursor.fetchall()
    #print('SQLite Version is {}'.format(result))
    
    ## get the inflection table as a list of words instead of tuple
    
    inflection_tuple = result[0][0]  # Get the first element of the first tuple
    inflection_words = inflection_tuple.split(':') 
    
    ##transliterate back the result to IAST for readability
    
    inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
    query_transliterateIAST = transliterateSLP1IAST(query_transliterate)
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]

    # Define row and column titles
    rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
    coltitles = ["Sg", "Du", "Pl"]

    #if indices:
        #print(f"The indices of '{query_transliterateIAST}' are {indices}.")
    #else:
        #print(f"'{query_transliterateIAST}' is not in the list.")

    from tabulate import tabulate

    # Your list of strings

    # Split the list into a 6x3 table
    table = [inflection_wordsIAST[i:i+3] for i in range(0, len(inflection_wordsIAST), 3)]

    # Highlight the matched cells with red color
    table = [["\033[31m" + cell + "\033[0m" if i*3 + j in indices else cell for j, cell in enumerate(row)] for i, row in enumerate(table)]

    # Add row titles to the table
    table = [[rowtitle] + row for rowtitle, row in zip(rowtitles, table)]

    # Print the table with column titles
    #print(tabulate(table, headers=coltitles, tablefmt="grid"))

    # Print the indices of the query in the list
    if indices:
        # Convert the indices to row and column names
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
       # print(f"The row and column names of '{query_transliterateIAST}' are {row_col_names}.")
   # else:
       # print(f"'{query_transliterateIAST}' is not in the list.")
    else: 
        row_col_names = None
        
    return [root_form_var, type_var, row_col_names, inflection_wordsIAST, original_word]






def SQLite_find_verb(verb):
    
    original_verb = verb
    
    # Transliterate input
    query_transliterate = anythingToSLP1(verb)
    
    # FormatSQL query
    query_builder = "SELECT * FROM vlgtab2 WHERE key = ?"
    
    ##OpenConnection, make SQL query to find the root
    
    sqliteConnection = sqlite3.connect('SQLite_db/vlgtab2.sqlite')

    cursor = sqliteConnection.cursor()
    #print('DB Init')

    #"select * from `$table` where `key`=\"$key\"";
    cursor.execute(query_builder, (query_transliterate,))
    result = cursor.fetchall()
    #print('SQLite Version is {}'.format(result))
    
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
            
    query_builder2 = "SELECT * FROM vlgtab1 WHERE stem = ?"

    sqliteConnection = sqlite3.connect('SQLite_db/vlgtab1.sqlite')

    cursor = sqliteConnection.cursor()
    #print('DB Init')

    cursor.execute(query_builder2, (root_form,))
    result = cursor.fetchall()
    #print('SQLite Version is {}'.format(result))

    selected_tuple = None

    # Iterate over the result list
    for model, stem, refs, data in result:
        if model == type_var:  # If the model matches type_var
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
    table = [inflection_wordsIAST[i:i+3] for i in range(0, len(inflection_wordsIAST), 3)]

    # Highlight the matched cells with red color
    table = [["\033[31m" + cell + "\033[0m" if i*3 + j in indices else cell for j, cell in enumerate(row)] for i, row in enumerate(table)]

    # Add row titles to the table
    table = [[rowtitle] + row for rowtitle, row in zip(rowtitles, table)]

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

        
    return [root_form_var, type_var, row_col_names, inflection_wordsIAST, original_verb]



## also map to the type.

import pandas as pd

# Read the Excel file into a DataFrame
type_map = pd.read_excel('resources/type_map.xlsx')

def root_any_word(word):
    result_root = SQLite_find_name(word)
    if not result_root:  # If process_word didn't find any results
        result_root = SQLite_find_verb(word)
    
    if result_root:
        # Get the second member of the tuple
        abbr = result_root[1]

        # Find the matching value in the 'abbr' column
        match = type_map[type_map['abbr'] == abbr]

        if not match.empty:
            description = match['description'].values[0]
            result_root = list(result_root)
            result_root[1] = description

    return result_root


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


def root_compounds(word):
    
    first_root = dict_word_iterative(word)
    first_root_entry = root_any_word(first_root[0])
    
    ## if it's a compound
    if first_root is not None and first_root[1] is not None and len(first_root[1]) >= 4:
        
        without_root = word.replace(first_root[0], '', 1)  # Only replace the first occurrence
        
        ##try the dictionary approach
        second_root = root_any_word(without_root)
        
        ##if the dictionary approach fails, try the iterative approach:
        if second_root == None:
            second_root = dict_word_iterative(without_root)
            second_root_try = root_any_word(second_root[0])
            
            if second_root_try is not None:
                second_root = second_root_try
            else: 
                second_root = second_root[0]
                      
            if second_root is not None:
                if first_root_entry is not None:
                    return [first_root_entry, second_root]
                else:
                    return [first_root[0], second_root]
            else:
                if first_root_entry is not None:
                    return [first_root_entry]
                else:
                    return [first_root[0]]
        else:
            if first_root_entry is not None:
                return [first_root_entry, second_root]
            else:
                return [first_root[0], second_root]
            
    ## if it's not a compound        
    else:
        if first_root_entry is not None:
            return [first_root_entry]
        else:
            return [first_root[0]]
            


def inflect(splitted_text):
    roots = []
    
#    print("splitted", splitted_text)
    for word in splitted_text: 
        root1 = root_any_word(word)
        #print("root1", root1)
        if root1 is not None:
            roots.append(root1)  
        else:
            compound_try = root_compounds(word)
            if compound_try is not None:
                roots.extend(compound_try)  
            else:
                roots.extend(word)  
                
    for i in range(len(roots)):
        if isinstance(roots[i], list):
            roots[i][0] = transliterateSLP1IAST(roots[i][0].replace('-', ''))
        else:
            roots[i] = transliterateSLP1IAST(roots[i].replace('-', ''))           
                
    return roots             


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
    
                for entry_dict in dictionary_json[word]:
                    dict_entry.append(entry_dict[4])           
                entry.append(key2)
                entry.append(dict_entry)
                entries.append(entry)
            
        elif isinstance(entry, str):
            
            dict_entry = []
            if entry in dictionary_json:  # Check if the key exists
                key2 = dictionary_json[entry][0][1]
    
                for entry_dict in dictionary_json[entry]:
                    dict_entry.append(entry_dict[4])    

                entry = [entry]    
                
                entry.append(key2)
                entry.append(dict_entry)
                entries.append(entry)
    return entries


# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,

def process(text):
    
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)
    

    inflections = inflect(splitted_text) 

    inflections_vocabulary = get_voc_entry(inflections)   
    

    return inflections_vocabulary


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



def preprocess(text):
    
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)

    print(splitted_text)


    splitted_text = remove_stopwords_list(splitted_text)    
    
    inflections = inflect(splitted_text) 
    
    entry_list = []
    for entry in inflections:
        entry_list.append(entry[0])
        inflections = entry_list    

    return inflections