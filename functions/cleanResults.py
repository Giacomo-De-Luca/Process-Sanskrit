from utils.lexicalResources import filtered_words
from utils.loadResources import mwdictionaryKeys
from functions.dictionaryLookup import get_voc_entry

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