import time
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.compoundAnalysis import root_compounds


prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'abhi', 'ni', 'pari', 'prati', 'parā', 'ava', 'adhi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duḥ']

def inflect(splitted_text, debug=False, session=None):
    roots = []
    
    i = 0
    while i < len(splitted_text):
        word = splitted_text[i]
        #print(f"Processing word: {word}")
        if word in prefixes and i + 1 < len(splitted_text):
            next_word = splitted_text[i + 1]

            if debug == True:
                print(f"Found prefix: {word}, next word: {next_word}")

            if word == 'sam':
                combined_words = ['sam' + next_word, 'saṃ' + next_word]
            elif word == 'vi':
                combined_words = ['vi' + next_word, 'vy' + next_word]
            else:
                combined_words = [word + next_word]

            rooted = None
            for combined_word in combined_words:
                if debug == True:
                    start_time = time.time()
                rooted = root_any_word(combined_word, session=session)
                
                if debug == True:
                    print(f"root_any_word({combined_word}) took {time.time() - start_time:.6f} seconds")
                if rooted is not None:
                    break  # Exit loop if a valid root is found

            if rooted is not None:
                roots.extend(rooted)
                i += 2  # Skip next word since it's part of the combined word
                continue
            else:
                if debug == True:
                    start_time = time.time()
                rooted_word = root_any_word(word, session=session)
                if debug == True:
                    print(f"root_any_word({word}) took {time.time() - start_time:.6f} seconds")
                if rooted_word is not None:
                    roots.extend(rooted_word)
                else:
                    if debug == True:
                        start_time = time.time()
                    compound_try = root_compounds(word, session=session)
                    if debug == True:
                        print(f"root_compounds({word}) took {time.time() - start_time:.6f} seconds")
                    if compound_try is not None:
                        roots.extend(compound_try)
                    else:
                        roots.append(word)
                i += 1  # Move to next word
        else:
            if debug == True:
                start_time = time.time()
            rooted = root_any_word(word, session=session)
            if debug == True:
                print(f"root_any_word({word}) took {time.time() - start_time:.6f} seconds")
            if rooted is not None:
                roots.extend(rooted)
            else:
                if debug == True:
                    start_time = time.time()
                compound_try = root_compounds(word, session=session)
                if debug == True:
                    print(f"root_compounds({word}) took {time.time() - start_time:.6f} seconds")
                if compound_try is not None:
                    roots.extend(compound_try)
                else:
                    roots.append(word)
            i += 1

    # Transliterate roots
    for j in range(len(roots)):
        if isinstance(roots[j], list):
            start_time = time.time()
            roots[j][0] = roots[j][0].replace('-', '')
        else:
            start_time = time.time()
            roots[j] = roots[j].replace('-', '')
    return roots