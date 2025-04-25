from typing import List, Dict, Tuple, Union
from process_sanskrit.utils.databaseSetup import Session, engine
from sqlalchemy import create_engine, text, Column, String
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import samMap


## add implementation to handle better both the cases of various spellings of M 
## and the handling of the cases of words ending with H --- 
## some dictionaries match entries with the nominative having a final H
## some keep the vocative that has no H
## there should be a map 
## if --- word ends with H 
## keep it in some dictionaries 
## and trim for others
## on the reverse 
## if the word ends with a vowel and the dictionary is one of those that search for H
## in that case add an H before the search 



def multidict(name: str, *args: str, source: str = "MW") -> Dict[str, Dict[str, List[str]]]:
    dict_names: List[str] = []
    dict_results: Dict[str, Dict[str, List[str]]] = {}
    name_component: str = ""
    
    # Collect dictionary names4
    if not args:
        dict_names.append(source)
    else:
        if "MW" in args:
            dict_names = ["MW"] + [x for x in args if x != "MW"]
        else:
            dict_names.extend(args)
    
    session = Session()
    
    # For each dictionary, perform queries and process results
    for dict_name in dict_names:
        
        # Initial query
        query_builder = f"""
        SELECT keys_iast, components, cleaned_body 
        FROM {dict_name} 
        WHERE keys_iast = :name 
        OR keys_iast LIKE :wildcard_name
        """
        wildcard_name = f"{name}"
        
        with engine.connect() as connection:
            results = connection.execute(
                text(query_builder), 
                {"name": name, "wildcard_name": wildcard_name}
            ).fetchall()

        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT keys_iast, components, cleaned_body FROM {dict_name} 
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
            SELECT keys_iast, components, cleaned_body FROM {dict_name} 
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



def consult_references(word: str, *dict_names: str) -> list[str, str, list[str]]:
    # Start with the dictionaries the user requested
    search_dictionaries = [*dict_names]

    # First check if the word exists in any of our specified dictionaries
    word_in_specified = any(d in DICTIONARY_REFERENCES.get(word, []) 
                          for d in search_dictionaries)

    # If word is not in our specified dictionaries but exists in others,
    # add those other dictionaries to our search list
    if not word_in_specified and word in DICTIONARY_REFERENCES:
        additional_dicts = DICTIONARY_REFERENCES[word]
        search_dictionaries.extend(additional_dicts)
        print(f"Word '{word}' not found in specified dictionaries. "
              f"Adding dictionaries: {additional_dicts}")

    # Now perform the search with either original or expanded dictionary list
    results = multidict(word, *search_dictionaries)
    
    if results[1]:  # If we found entries
        return results
        
    return [word, word, [word]]  # Default format if no results found




def get_voc_entry(list_of_entries, *args, source: str = "mw"):

    dict_names: List[str] = [*args]


    # Collect dictionary names
    if not dict_names:
        dict_names.append(source)

    #print("dict_names", dict_names)

    #print("list_of_entries", list_of_entries)
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            #print("entry", entry)

            word = entry[0]

            if '*' not in word and '_' not in word and '%' not in word:

                if word in DICTIONARY_REFERENCES.keys():  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi                    
                    entry = entry + consult_references(word, *dict_names)

                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry

                entries.append(entry)
            else:
                entry = entry + consult_references(word, *dict_names)
                entries.append(entry)
            
        elif isinstance(entry, str):
            
            if '*' not in entry and '_' not in entry and '%' not in entry:
                if entry in DICTIONARY_REFERENCES.keys():  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    entry = [entry] + consult_references(entry, *dict_names)

                elif entry[:3] in samMap:
                        tentative = samMap[entry[:3]] + entry[3:]
                        if tentative in DICTIONARY_REFERENCES.keys():
                            entry = [entry] + consult_references(tentative, *dict_names)
                        else:
                            entry = [entry, entry, [entry]]

                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                entries.append(entry)
            else:
                entry = [entry] + consult_references(entry, *dict_names)
                entries.append(entry)
    return entries




def get_mwword(word:str)->list[str, str, list[str]] : 
        session = Session()
        query_builder = text("SELECT components, cleaned_body FROM mwclean WHERE keys_iast = :word")
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]

import time