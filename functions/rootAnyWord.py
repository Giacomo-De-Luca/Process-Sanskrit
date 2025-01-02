import time 
from functions.SQLiteFind import SQLite_find_name, SQLite_find_verb
from utils.loadResources import type_map
from utils.lexicalResources import variableSandhi

##given a name finds the root

def root_any_word(word, attempted_words=None, timed=True):
    if attempted_words is None:
        attempted_words = set()

    # If the word has already been attempted, return None to avoid infinite loop
    if word in attempted_words:
        return None

    # Add the current word to the set of attempted words
    attempted_words.add(word)

    if timed:
        start_time = time.time()
    result_roots_name = SQLite_find_name(word)
    if timed:
        print(f"SQLite_find_name({word}) took {time.time() - start_time:.6f} seconds")

    if timed:
        start_time = time.time()
    result_roots_verb = SQLite_find_verb(word)
    if timed:
        print(f"SQLite_find_verb({word}) took {time.time() - start_time:.6f} seconds")

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
            if timed:
                start_time = time.time()
            # Find the matching value in the 'abbr' column
            match = type_map[type_map['abbr'] == abbr]
            if timed:
                print(f"Matching abbr {abbr} took {time.time() - start_time:.6f} seconds")
            
            if not match.empty:
                description = match['description'].values[0]
                result[1] = description
                result_roots[i] = result
        return result_roots

    # If no result is found, try replacements based on variableSandhi
    if word[-1] in variableSandhi:
        for replacement in variableSandhi[word[-1]]:
            tentative = word[:-1] + replacement
            if timed:
                start_time = time.time()
            attempt = root_any_word(tentative, attempted_words, timed)
            if timed:
                print(f"root_any_word({tentative}) took {time.time() - start_time:.6f} seconds")
            if attempt is not None:
                return attempt
    
    samMap = {
        'sam': 'saṃ',
        'saṃ': 'sam',
        'saṅ': 'saṃ',
        'san': 'saṃ',
        'sañ': 'saṃ',
    }

    # Different spellings for sam, - it is so common that it deserves its own rule
    if word[0:3] in samMap:
        print("tentative", word)
        tentative = samMap[word[0:3]] + word[3:]
        if timed:
            start_time = time.time()
        attempt = root_any_word(tentative, attempted_words, timed)
        if timed:
            print(f"root_any_word({tentative}) took {time.time() - start_time:.6f} seconds")
        print("attempt", attempt)
        if attempt is not None:
            return attempt

    return None

