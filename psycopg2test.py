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
from sqlalchemy import create_engine, text
from typing import List, Dict, Any
import psycopg2
import os 
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL = os.environ['DATABASE_URL']



logging.basicConfig(level=logging.CRITICAL)

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






parser = Parser(output_encoding='slp1')

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

##given a name finds the root

def SQLite_find_name(name):

 

    def query1(word):


        query_builder = "SELECT * FROM lgtab2 WHERE key = ?"  

        try:
            connection = psycopg2.connect(
                user="jack",  # Replace with your PostgreSQL username
                password="again",  # Replace with your PostgreSQL password
                host="localhost",  # Or the appropriate host if remote
                port="5432",  # PostgreSQL port (default is 5432)
                database="sanskritmagicdb"  # Replace with your database name
            )

            cursor = connection.cursor()

            # Example query execution
            cursor.execute(query_builder, (word,))
            results = cursor.fetchall()


        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")

        return results
    
    results = query1(name)
    
    if not results:  # If query1 didn't find any results
        if name[-1] == 'M':
            name = name[:-1] + 'm'
            results = query1(name)
    
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
        query_transliterateIAST = transliterateSLP1IAST(name)

        ##make Inflection Table

        indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]
        # Define row and column titles
        rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
        coltitles = ["Sg", "Du", "Pl"]



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
        
        outcome.append([word_refs, type_var, row_col_names, inflection_wordsIAST, name])
    return outcome



def SQLite_find_verb(verb):
    
    root_form = None

    
    # FormatSQL query
    query_builder = "SELECT * FROM vlgtab2 WHERE key = ?"
    
    ##OpenConnection, make SQL query to find the root
    
    sqliteConnection = sqlite3.connect('resources/SQLite_db/vlgtab2.sqlite')
    cursor = sqliteConnection.cursor()
    #"select * from `$table` where `key`=\"$key\"";
    cursor.execute(query_builder, (verb,))
    result = cursor.fetchall()
    sqliteConnection.close()
    #print("result1", result)

    
    
    for inflected_form, type, root_form in result:

        if not root_form:  # If root_form is None or empty
            return  # End the function
        type_var = type
    
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
    query_transliterateIAST = transliterateSLP1IAST(verb)
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]

    # Define row and column titles
    rowtitles = ["First", "Second", "Third"]
    coltitles = ["Sg", "Du", "Pl"]


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
        
    return [[stem, type_var, row_col_names, inflection_wordsIAST, verb]]



## also map to the type.
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
            


def inflect(splitted_text):
    roots = []
    
#    print("splitted", splitted_text)
    for word in splitted_text: 
        
        rooted = root_any_word(word)
        if rooted is not None:
            for root in rooted:
                roots.append(root)  
        else:
            print("here breaks", word)
            compound_try = root_compounds(word)
            if compound_try is not None:
                roots.extend(compound_try)  
                #print("compound_try", compound_try)
            else:
                roots.extend(word)  
                
    for i in range(len(roots)):
        if isinstance(roots[i], list):
            #print("debug", roots)
            print(roots[i][0])
            roots[i][0] = transliterateSLP1IAST(roots[i][0].replace('-', ''))
        else:
            print(roots[i][0])
            roots[i] = transliterateSLP1IAST(roots[i].replace('-', ''))           
    #print("inflect roots", roots)
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



def process(text):

    if ' ' not in text:
        #print("single_word", text)
        transliterated_text = anythingToSLP1(text)     
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
        elif list_of_entries[i][0] == "anu":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "anu"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["anu" + list_of_entries[j][0]])
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

path = "/resources/Sanskrit_dictionaries/"

def multidict(name: str, *args: str, source: str = "MW") -> Dict[str, List[Any]]:
    
    dict_names: List[str] = []
    dict_results: Dict[str, List[Any]] = {}
    
    # Check if any arguments were provided
    if not args:
        dict_names.append(source)  # If no args, use the source (default is "MW")
    else:
        for dict_name in args:
            dict_names.append(dict_name)
    
    # For each dictionary name, build the path and query
    for dict_name in dict_names:
        path_builder = "sqlite:///" + path + dict_name + ".sqlite"
        
        # Create SQLAlchemy engine
        engine = create_engine(path_builder)
        
        query_builder = f"""
        SELECT data FROM {dict_name} 
        WHERE key = :name 
        OR key LIKE :wildcard_name
        """

        # Prepare the wildcard by taking the word minus the last letter and appending '_'
        wildcard_name = f"{name[:-1]}_"

        # Execute the query with both the exact match and the wildcard condition
        with engine.connect() as connection:
            results = connection.execute(
                text(query_builder), 
                {"name": name, "wildcard_name": wildcard_name}
            ).fetchall()


        # Step 2 and Step 3 combined: If no exact match, search for name + _ and name minus last letter + _
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT data FROM {dict_name} 
            WHERE key LIKE :name1 
            OR key LIKE :name2
            """
            with engine.connect() as connection:
                results = connection.execute(text(query_builder), {"name1": name + "_", "name2": name[:-1] + "_"}).fetchall()

        # Add the results to the dict_results with the dictionary name as the key
        dict_results[dict_name] = [dict(row) for row in results]
    
    return dict_results  # Return the dictionary containing all results

# Example usage
#results = multidict("yoga", "MW" "AP90")
#print(results)