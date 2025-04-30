from process_sanskrit.utils.lexicalResources import filtered_words
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES
from process_sanskrit.functions.dictionaryLookup import dict_search
import re
import regex

def extract_roots(list_of_entries, debug=False):
    roots = []
    i = 0
    
    while i < len(list_of_entries):
        current_entry = list_of_entries[i]
        
        # If entry has element at index 4 (original word)
        if len(current_entry) > 4:
            original_word = current_entry[4]
            stemmed_forms = []
            
            # Collect all stemmed forms for this original word
            j = i
            while j < len(list_of_entries) and len(list_of_entries[j]) > 4 and list_of_entries[j][4] == original_word:
                if list_of_entries[j][0] not in stemmed_forms:  # Avoid duplicates
                    stemmed_forms.append(list_of_entries[j][0])
                j += 1
            
            # Add as tuple if multiple stems, otherwise as single string
            if len(stemmed_forms) > 1:
                if debug:
                    print(f"Multiple stems for '{original_word}': {stemmed_forms}")
                roots.append(tuple(stemmed_forms))
            else:
                if debug:
                    print(stemmed_forms[0])
                roots.append(stemmed_forms[0])
                
            i = j  # Skip already processed entries
        else:
            # Handle entries without entry[4]
            if not roots or (isinstance(roots[-1], str) and roots[-1] != current_entry[0]) or \
               (isinstance(roots[-1], tuple) and current_entry[0] not in roots[-1]):
                if debug:
                    print(current_entry[0])
                roots.append(current_entry[0])
            i += 1
    
    return roots

def roots_splitted(list_of_entries, debug=False):

        root_dict = {}
        separators = r"[-—,/]"
        for entry in list_of_entries:
            if len(entry) == 7:
                parts = re.split(separators, entry[5])
                parts = [regex.sub(r'[^\p{L}]', '', part) for part in parts if part]
                parts = list(dict.fromkeys(parts))  # Remove duplicates while preserving order
                if entry[0] not in root_dict:
                    root_dict[entry[0]] = parts
            elif len(entry) == 3:
                parts = re.split(separators, entry[1])
                parts = [regex.sub(r'[^\p{L}]', '', part) for part in parts if part]
                parts = list(dict.fromkeys(parts))  # Remove duplicates while preserving order
                if entry[0] not in root_dict:
                    root_dict[entry[0]] = parts
        return root_dict
    

def clean_results(list_of_entries, mode="detailed", debug=False):

    i = 0
   
    #print("is it broken here?", list_of_entries)

    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        # Check if the word is in filtered_words
        if list_of_entries[i][0] in filtered_words:
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == list_of_entries[i][0]:
                del list_of_entries[i + 1]

        ### make a new rule for api ---> if word is api and the next entries are apya and ap remove them directly

        if list_of_entries[i][0] == "duḥ" and list_of_entries[i+1][0] == "kha":
            replacement = dict_search(["duḥkha"])
            if replacement is not None:
                list_of_entries[i] = replacement[0]
                del list_of_entries[i + 1]
                if list_of_entries[1+2] == "kha":
                    del list_of_entries[i + 2]  ##it's kha also as well

        if len(list_of_entries[i]) >= 5 and list_of_entries[i][0][-1] == "n" and list_of_entries[i][4] != list_of_entries[i][0]:
            #print("the one not replaced:", list_of_entries[i])
            if list_of_entries[i][4] in DICTIONARY_REFERENCES.keys():
                replacement = dict_search([list_of_entries[i][4]])
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
                
                voc_entry = None
                if list_of_entries[j][0] not in SANSKRIT_PREFIXES:
                    voc_entry = dict_search(["sam" + list_of_entries[j][0]])
                #print("voc_entry", voc_entry)

                ##non ha senso
                ## TODO replace this entirely
                
                # Ensure voc_entry is not None and has the expected structure
                if (voc_entry is not None and len(voc_entry) > 0 and 
                    isinstance(voc_entry[0], list) and len(voc_entry[0]) > 2 and 
                    isinstance(voc_entry[0][2], dict) and 'MW' in voc_entry[0][2]):
                    
                    # Check if the first key of the dictionary inside MW matches the condition
                    first_key = next(iter(voc_entry[0][2]['MW']), None)
                    if first_key and voc_entry[0][0] == first_key:
                        #print("revised query", ["saṃ" + list_of_entries[j][0]])
                        voc_entry = dict_search("saṃ" + list_of_entries[j][0])
                        #print("revise_voc_entry", voc_entry)
        
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "anu"
        if list_of_entries[i][0] == "anu":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "anu"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = dict_search(["anu" + list_of_entries[j][0]])
                ## testing to see if the check works. 
                print(voc_entry)
                if voc_entry[0][0] != voc_entry[0][2][0]:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "ava"
        if list_of_entries[i][0] == "ava":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "ava"):
                j += 1
            if j < len(list_of_entries):
                #print("testing with:", ["ava" + list_of_entries[j + 1][0]])
                
                voc_entry = dict_search(["ava" + list_of_entries[j + 1][0]])
                if voc_entry[0][0] != voc_entry[0][2][0]:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]        
        i += 1  
    

    if mode == "parts":
            return roots_splitted(list_of_entries, debug=debug)
    elif mode == "roots":
            return extract_roots(list_of_entries, debug=debug)
    else:  # Default case when roots is "none" or any other value
        return list_of_entries
