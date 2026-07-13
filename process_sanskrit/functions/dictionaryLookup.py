import re
from typing import List, Dict

from sqlalchemy import text
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import samMap


## The dictionary consulted when the caller names none.  consult_references()
## widens the search anyway if the word is not in it, so this is a starting
## point rather than a limit.
DEFAULT_DICTIONARY = "mw"


def _dictionary_table(dictionary_name: str) -> str:
    table_name = dictionary_name.lower()
    if re.fullmatch(r"[a-z_][a-z0-9_]*", table_name) is None:
        raise ValueError(f"Invalid dictionary table name: {dictionary_name}")
    return table_name


def _glob_literal(value: str) -> str:
    """Escape SQLite GLOB metacharacters in an internally generated prefix."""
    return value.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


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


def multidict(name: str, *args: str, source: str = "MW", session=None) -> Dict[str, Dict[str, List[str]]]:
    dict_names: List[str] = []
    dict_results: Dict[str, Dict[str, List[str]]] = {}
    name_component: str = ""
    fallback_headword: str = ""
    
    # Collect dictionary names4
    if not args:
        dict_names.append(source)
    else:
        if "MW" in args:
            dict_names = ["MW"] + [x for x in args if x != "MW"]
        else:
            dict_names.extend(args)
        
    # For each dictionary, perform queries and process results
    for dict_name in dict_names:
        table_name = _dictionary_table(dict_name)

        if "_" in name or "%" in name:
            # Preserve explicit SQL wildcard behavior. Leading wildcards cannot
            # use a B-tree index and intentionally remain the uncommon slow path.
            query_builder = text(
                f"SELECT keys_iast, components, cleaned_body "
                f"FROM {table_name} WHERE keys_iast LIKE :pattern"
            )
            results = session.execute(query_builder, {"pattern": name}).fetchall()
        else:
            # Equality is the overwhelmingly common path. Keeping LIKE out of
            # this statement lets SQLite use the existing keys_iast index.
            query_builder = text(
                f"SELECT keys_iast, components, cleaned_body "
                f"FROM {table_name} WHERE keys_iast = :name"
            )
            results = session.execute(query_builder, {"name": name}).fetchall()

        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = text(
                f"SELECT keys_iast, components, cleaned_body "
                f"FROM {table_name} WHERE keys_iast GLOB :pattern"
            )
            pattern = _glob_literal(name[:-1]) + "?"
            results = session.execute(
                query_builder,
                {"pattern": pattern},
            ).fetchall()
        
        #print(f"Results for {dict_name}: {results}")
        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = text(
                f"SELECT keys_iast, components, cleaned_body "
                f"FROM {table_name} "
                "WHERE keys_iast GLOB :name1 OR keys_iast GLOB :name2"
            )
            results = session.execute(
                query_builder,
                {
                    "name1": _glob_literal(name) + "?",
                    "name2": _glob_literal(name[:-1]) + "?",
                },
            ).fetchall()

        #print(f"Results for {dict_name} after second query: {results}")
        
        # Group results by components
        component_dict: Dict[str, List[str]] = {}
        for row in results:
            #print(f"Row: {row}")
            key_iast, components, cleaned_body = row
            if not fallback_headword:
                fallback_headword = key_iast
            if not name_component and components:
                name_component = components
            #print(f"key_iast: {key_iast}, components: {components}, cleaned_body: {cleaned_body}")
            if key_iast in component_dict:
                component_dict[key_iast].append(cleaned_body)
            else:
                component_dict[key_iast] = [cleaned_body]        
        # Add to dict_results
        dict_results[dict_name] = component_dict
    
    # Some source dictionaries leave ``components`` NULL even for a real
    # headword.  Downstream ``parts`` output needs a string in this slot, so use
    # the matched headword when no dictionary supplied a decomposition.
    return [name_component or fallback_headword, dict_results]



# Example usage
#results = multidict("yoga", "MW" "AP90")
#print(results)



def consult_references(word: str, *dict_names: str, session=None) -> list[str, str, list[str]]:
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
        #print(f"Word '{word}' not found in specified dictionaries. "
        #      f"Adding dictionaries: {additional_dicts}")

    # Now perform the search with either original or expanded dictionary list
    results = multidict(word, *search_dictionaries, session=session)
    
    if results[1]:  # If we found entries
        return results
        
    return [word, word, [word]]  # Default format if no results found




def dict_search(list_of_entries, *args, source: str = DEFAULT_DICTIONARY, session=None):
    """
    Get vocabulary entries for a list of words.
    
    Args:
        list_of_entries: Words to look up
        *args: Dictionary names to search
        source: Default dictionary
        session: SQLAlchemy session to use (improves performance)
    """

    #print("list_of_entries", list_of_entries)
    # Create session at the top level if not provided
    close_session = False
    if session is None:
        from process_sanskrit.utils.databaseSetup import get_session
        session = get_session()
        close_session = True
    
    try:
        dict_names: List[str] = [*args]

        # Collect dictionary names
        if not dict_names:
            dict_names.append(source)

        entries = []
        for entry in list_of_entries:        
            if isinstance(entry, list):
                word = entry[0]

                if '*' not in word and '_' not in word and '%' not in word:
                    if word in DICTIONARY_REFERENCES:
                        entry = entry + consult_references(word, *dict_names, session=session)
                    else:
                        entry = [entry, entry, [entry]]
                    entries.append(entry)
                else:
                    entry = entry + consult_references(word, *dict_names, session=session)
                    entries.append(entry)
                
            elif isinstance(entry, str):
                if '*' not in entry and '_' not in entry and '%' not in entry:
                    if entry in DICTIONARY_REFERENCES:
                        entry = [entry] + consult_references(entry, *dict_names, session=session)
                    elif entry[:3] in samMap:
                        tentative = samMap[entry[:3]] + entry[3:]
                        if tentative in DICTIONARY_REFERENCES:
                            entry = [entry] + consult_references(tentative, *dict_names, session=session)
                        else:
                            entry = [entry, entry, [entry]]
                    else:
                        entry = [entry, entry, [entry]]
                    entries.append(entry)
                else:
                    entry = [entry] + consult_references(entry, *dict_names, session=session)
                    entries.append(entry)
        
        return entries
    finally:
        if close_session:
            session.close()
