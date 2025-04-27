import re
import regex
from sqlalchemy.sql import text


def SQLite_find_name(name, session=None):

    outcome = []    

    def query1(word):

        try:
            query_builder = text("SELECT * FROM lgtab2 WHERE key = :word")
            results = session.execute(query_builder, {'word': word}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []
        return results

    results = query1(name)
    
    if not results:  # If query1 didn't find any results
        if name[-1] == 'ṃ':
            name = name[:-1] + 'm'
            results = query1(name)
        elif name[-1] == 'm':
            name = name[:-1] + 'ṃ'
            results = query1(name)
    
    for inflected_form, type, root_form in results: 
        if not root_form:  # If root_form is None or empty
            return  # End the function

        def query2(root_form: str, type: str):

            try:
                query_builder2 = text("SELECT * FROM lgtab1 WHERE stem = :root_form and model = :type ")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []
            return results
        
        result = query2(root_form, type)
        word_refs = regex.findall(r",(\p{L}+)", result[0][2])[0]

        inflection_tuple = result[0][3]  # Get the first element of the first tuple
        inflection_words = inflection_tuple.split(':') 

        ##make Inflection Table
        indices = [i for i, x in enumerate(inflection_words) if x == name]
        rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
        coltitles = ["Sg", "Du", "Pl"]

        if indices:
            row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
        else: 
            row_col_names = None
        outcome.append([word_refs, type, row_col_names, inflection_words, name])

    return outcome



def SQLite_find_verb(verb, session=None):
    
    root_form = None

    def query1(verb):

        try:
            query_builder = text("SELECT * FROM vlgtab2 WHERE key = :verb")
            results = session.execute(query_builder, {'verb': verb}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []

        return results

    result = query1(verb)
    
    for inflected_form, type, root_form in result:

        if not root_form:  # If root_form is None or empty
            return  # End the function
        type_var = type

        def query2(root_form: str, type: str):

            try:
                query_builder2 = text("SELECT * FROM vlgtab1 WHERE stem = :root_form and model = :type")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []

            return results
        
        result = query2(root_form, type)
    
    selected_tuple = None

    # Iterate over the result list
    for model, stem, refs, data in result:
        if model == type_var:  # If the model matches type_var
            ref_word = regex.search(r",(\p{L}+)", refs).group(1)
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
    
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_words) if x == verb]

    # Define row and column titles
    rowtitles = ["First", "Second", "Third"]
    coltitles = ["Sg", "Du", "Pl"]


    if indices:
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
    else:
        row_col_names = None
        
    return [[stem, type_var, row_col_names, inflection_words, verb]]



def optimized_find_name(name, session=None):
    """
    Optimized version of SQLite_find_name using a single JOIN query.
    Queries lgtab2 and lgtab1 tables to find inflection data for a given word.
    
    Args:
        name (str): The word to look up in IAST format
        
    Returns:
        list: List of results containing [word_refs, model, row_col_names, inflection_words, name]
    """
    outcome = []
    
    try:
        # Single query using JOIN to get all required data at once
        query = """
        SELECT 
            t2.key,    -- Original inflected form
            t2.model,  -- Morphological model/type
            t2.stem,   -- Root/stem form
            t1.data    -- Inflection data string
        FROM lgtab2 t2
        JOIN lgtab1 t1 ON t1.stem = t2.stem AND t1.model = t2.model
        WHERE t2.key = :word
        """
        
        # Try with original name
        results = session.execute(text(query), {'word': name}).fetchall()
        
        # If no results and name ends with ṃ/m, try alternative
        if not results:
            if name[-1] == 'ṃ':
                alt_name = name[:-1] + 'm'
                results = session.execute(text(query), {'word': alt_name}).fetchall()
            elif name[-1] == 'm':
                alt_name = name[:-1] + 'ṃ'
                results = session.execute(text(query), {'word': alt_name}).fetchall()
        
        # Process results
        for row in results:
            key, model, stem, inflection_data = row
            
            if not stem:  # Skip if no stem found
                continue
                
            # Split inflection data into words
            inflection_words = inflection_data.split(':')
            
            # Generate inflection table data
            indices = [i for i, x in enumerate(inflection_words) if x == name]
            rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
            coltitles = ["Sg", "Du", "Pl"]
            
            row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices] if indices else None
            
            # Add to results using the same structure as the original function
            outcome.append([stem, model, row_col_names, inflection_words, name])
            
    except Exception as error:
        print(f"Error in database query: {error}")

        
    return outcome

def optimized_find_verb(verb, session=None):
    """
    Optimized version of SQLite_find_verb using a single JOIN query.
    Queries vlgtab2 and vlgtab1 tables to find conjugation data for a given verb.
    
    Args:
        verb (str): The verb to look up in IAST format
        
    Returns:
        list: List containing [stem, model, row_col_names, inflection_words, verb]
    """
    try:
        query = """
        SELECT 
            t2.key,    -- Original inflected form
            t2.model,  -- Verb model/type
            t2.stem,   -- Root/stem form
            t1.data    -- Conjugation data string
        FROM vlgtab2 t2
        JOIN vlgtab1 t1 ON t1.stem = t2.stem AND t1.model = t2.model
        WHERE t2.key = :verb
        """
        
        results = session.execute(text(query), {'verb': verb}).fetchall()
        
        if not results:
            return None
            
        # Process first matching result
        key, model, stem, inflection_data = results[0]
        
        if not stem:
            return None
            
        # Split inflection data into words
        inflection_words = inflection_data.split(':')
        
        # Generate conjugation table data
        indices = [i for i, x in enumerate(inflection_words) if x == verb]
        rowtitles = ["First", "Second", "Third"]
        coltitles = ["Sg", "Du", "Pl"]
        
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices] if indices else None
        
        # Return with same structure as original function
        return [[stem, model, row_col_names, inflection_words, verb]]
        
    except Exception as error:
        print(f"Error in database query: {error}")
        return None
