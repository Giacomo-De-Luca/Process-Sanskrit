from typing import List, Dict, Tuple, Union
from utils.databaseSetup import Session, engine
from sqlalchemy import create_engine, text, Column, String
from utils.loadResources import mwdictionaryKeys




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





def get_mwword(word:str)->list[str, str, list[str]] : 
        session = Session()
        query_builder = text("SELECT components, cleaned_body FROM mwclean WHERE keys_iast = :word")
        print("query_builder", query_builder, word)
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]

import time

def get_voc_entry(list_of_entries, *dict_names):
    print("list_of_entries", list_of_entries)
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]
            print("word that should be matched", word)

            if '*' not in word and '_' not in word and '%' not in word:

                start_time = time.time() 
                if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    print("word matched", word)
                    end_time = time.time() - start_time
                    print("just list lookup", end_time)
                    entry = entry + multidict(word, *dict_names)
                    end_time = time.time() - start_time
                    print("list + multidict", end_time)

                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                    end_time = time.time() - start_time
                    print("end_time", end_time)
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


